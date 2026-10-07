from eth_consensus_specs.test.context import always_bls, spec_state_test, with_gloas_and_later
from eth_consensus_specs.test.helpers.gossip import (
    get_filename,
    get_seen,
    get_store_from_state,
    run_validate_gossip,
)
from eth_consensus_specs.test.helpers.proposer_slashings import (
    get_valid_proposer_slashing,
    prepare_process_proposer_slashing,
)


@with_gloas_and_later
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__cancellation_after_slashing(spec, state):
    slashing, proposer_index = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        parent_root_2=b"\x99" * 32,
        proposer_slashed=True,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
    )
    # Two payments for the same validator, in the previous and current epochs.
    state.builder_pending_payments[spec.SLOTS_PER_EPOCH - 1] = state.builder_pending_payments[
        spec.SLOTS_PER_EPOCH
    ].copy()
    earlier = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=proposer_index,
        slot=state.slot - 1,
        signed_1=True,
        signed_2=True,
    )
    invalid_1 = slashing.copy()
    invalid_1.signed_header_1.signature = spec.BLSSignature()
    invalid_2 = slashing.copy()
    invalid_2.signed_header_2.signature = spec.BLSSignature()

    yield "topic", "meta", "proposer_slashing"
    yield "state", state
    store, anchor = get_store_from_state(spec, state)
    yield get_filename(anchor), anchor
    yield "blocks", "meta", [{"block": get_filename(anchor)}]
    seen = get_seen(spec)
    messages = []
    # Invalid proofs must not poison the cache; a proof for another slot must
    # not suppress cancellation, and duplicate cancellation proofs are ignored.
    cases = [
        (invalid_1, "reject", "invalid proposer slashing signature"),
        (invalid_2, "reject", "invalid proposer slashing signature"),
        (earlier, "valid", None),
        (slashing, "valid", None),
        (slashing, "ignore", "already seen proposer slashing for this proposal"),
    ]
    for proof in (invalid_1, invalid_2, earlier, slashing):
        yield get_filename(proof), proof
    for proof, expected, expected_reason in cases:
        result, reason = run_validate_gossip(spec, seen=seen, store=store, proposer_slashing=proof)
        assert (result, reason) == (expected, expected_reason)
        message = {"message": get_filename(proof), "expected": expected}
        if reason is not None:
            message["reason"] = reason
        messages.append(message)
    yield "messages", "meta", messages
