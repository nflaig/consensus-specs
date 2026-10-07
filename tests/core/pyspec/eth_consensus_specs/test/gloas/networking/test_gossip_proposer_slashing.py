from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_gloas_and_later,
)
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


def _prepare_cancellation(spec, state, previous_epoch=False, **kwargs):
    return prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=-spec.SLOTS_PER_EPOCH if previous_epoch else 0,
        parent_root_2=b"\x99" * 32,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        **kwargs,
    )


def _run_messages(spec, state, slashings, expected_results):
    yield "topic", "meta", "proposer_slashing"
    yield "state", state
    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    seen = get_seen(spec)
    messages = []
    yielded = set()
    for slashing, (expected, expected_reason) in zip(slashings, expected_results, strict=True):
        filename = get_filename(slashing)
        if filename not in yielded:
            yield filename, slashing
            yielded.add(filename)
        result, reason = run_validate_gossip(
            spec, seen=seen, store=store, proposer_slashing=slashing
        )
        assert result == expected
        assert reason == expected_reason
        message = {"message": filename, "expected": expected}
        if reason is not None:
            message["reason"] = reason
        messages.append(message)
    yield "messages", "meta", messages


@with_gloas_and_later
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__cancellation_already_slashed_current_epoch(spec, state):
    slashing, _ = _prepare_cancellation(spec, state, proposer_slashed=True)
    yield from _run_messages(spec, state, [slashing], [("valid", None)])


@with_gloas_and_later
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__cancellation_already_slashed_previous_epoch(spec, state):
    slashing, _ = _prepare_cancellation(spec, state, previous_epoch=True, proposer_slashed=True)
    yield from _run_messages(spec, state, [slashing], [("valid", None)])


@with_gloas_and_later
@spec_state_test
def test_gossip_proposer_slashing__cancellation_after_seen_other_slot(spec, state):
    slashing, proposer_index = _prepare_cancellation(spec, state)
    earlier = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=proposer_index,
        slot=state.slot - 1,
        signed_1=True,
        signed_2=True,
    )
    yield from _run_messages(spec, state, [earlier, slashing], [("valid", None), ("valid", None)])


@with_gloas_and_later
@spec_state_test
def test_gossip_proposer_slashing__cancellations_for_different_slots(spec, state):
    slashing, proposer_index = _prepare_cancellation(spec, state, proposer_slashed=True)
    previous_slot = state.slot - 1
    previous_index = previous_slot % spec.SLOTS_PER_EPOCH
    state.builder_pending_payments[previous_index] = state.builder_pending_payments[
        spec.SLOTS_PER_EPOCH
    ].copy()
    earlier = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=proposer_index,
        slot=previous_slot,
        signed_1=True,
        signed_2=True,
    )
    yield from _run_messages(spec, state, [earlier, slashing], [("valid", None), ("valid", None)])


@with_gloas_and_later
@spec_state_test
def test_gossip_proposer_slashing__ignore_duplicate_cancellation(spec, state):
    slashing, _ = _prepare_cancellation(spec, state, proposer_slashed=True)
    yield from _run_messages(
        spec,
        state,
        [slashing, slashing],
        [("valid", None), ("ignore", "already seen proposer slashing for this proposal")],
    )


@with_gloas_and_later
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__reject_cancellation_invalid_signature_1(spec, state):
    slashing, proposer_index = _prepare_cancellation(
        spec, state, proposer_slashed=True, signed_1=False
    )
    valid_slashing = get_valid_proposer_slashing(
        spec, state, slashed_index=proposer_index, signed_1=True, signed_2=True
    )
    yield from _run_messages(
        spec,
        state,
        [slashing, valid_slashing],
        [("reject", "invalid proposer slashing signature"), ("valid", None)],
    )


@with_gloas_and_later
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__reject_cancellation_invalid_signature_2(spec, state):
    slashing, proposer_index = _prepare_cancellation(
        spec, state, proposer_slashed=True, signed_2=False
    )
    valid_slashing = get_valid_proposer_slashing(
        spec, state, slashed_index=proposer_index, signed_1=True, signed_2=True
    )
    yield from _run_messages(
        spec,
        state,
        [slashing, valid_slashing],
        [("reject", "invalid proposer slashing signature"), ("valid", None)],
    )


@with_gloas_and_later
@spec_state_test
def test_gossip_proposer_slashing__reject_cancellation_foreign_proposer(spec, state):
    slashing, proposer_index = _prepare_cancellation(spec, state, proposer_slashed=True)
    state.builder_pending_payments[spec.SLOTS_PER_EPOCH].proposer_index = (
        proposer_index + 1
    ) % len(state.validators)
    yield from _run_messages(spec, state, [slashing], [("reject", "proposer is not slashable")])


@with_gloas_and_later
@spec_state_test
def test_gossip_proposer_slashing__reject_cancellation_zero_amount(spec, state):
    slashing, _ = _prepare_cancellation(spec, state, proposer_index=0, proposer_slashed=True)
    state.builder_pending_payments[spec.SLOTS_PER_EPOCH].withdrawal.amount = 0
    state.builder_pending_payments[spec.SLOTS_PER_EPOCH].weight = 1000
    yield from _run_messages(spec, state, [slashing], [("reject", "proposer is not slashable")])


@with_gloas_and_later
@spec_state_test
def test_gossip_proposer_slashing__reject_cancellation_outside_window(spec, state):
    _, proposer_index = _prepare_cancellation(spec, state, proposer_slashed=True)
    slashing = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=proposer_index,
        slot=state.slot - 2 * spec.SLOTS_PER_EPOCH,
        signed_1=True,
        signed_2=True,
    )
    yield from _run_messages(spec, state, [slashing], [("reject", "proposer is not slashable")])
