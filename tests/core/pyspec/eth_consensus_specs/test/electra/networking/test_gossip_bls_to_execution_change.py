from eth_consensus_specs.test.context import (
    spec_configured_state_test,
    with_all_phases_from_to,
)
from eth_consensus_specs.test.helpers.bls_to_execution_changes import (
    get_signed_address_change as get_signed_bls_to_execution_change,
)
from eth_consensus_specs.test.helpers.constants import ELECTRA, HEZE
from eth_consensus_specs.test.helpers.deposits import prepare_pending_deposit
from eth_consensus_specs.test.helpers.gossip import (
    get_filename,
    get_seen,
    get_store_from_state,
    run_validate_gossip,
)
from eth_consensus_specs.test.helpers.keys import pubkeys


# Heze does not create validators with BLS withdrawal credentials from deposits
@with_all_phases_from_to(ELECTRA, HEZE)
@spec_configured_state_test({}, activate_at_genesis=True)
def test_gossip_bls_to_execution_change__valid_validator_added_at_epoch_boundary(spec, state):
    """
    Test that a `bls_to_execution_change` is valid for a validator that the epoch
    transition adds from a pending deposit while the head state is still in the
    previous epoch.
    """
    yield "topic", "meta", "bls_to_execution_change"

    # Queue a deposit for a new validator that the next epoch transition applies
    validator_index = len(state.validators)
    pending_deposit = prepare_pending_deposit(
        spec, validator_index, spec.MIN_ACTIVATION_BALANCE, signed=True
    )
    state.pending_deposits.append(pending_deposit)
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]

    # The current slot is the first slot of the next epoch, still without a block
    next_epoch_start_slot = spec.compute_start_slot_at_epoch(spec.get_current_epoch(state) + 1)
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, next_epoch_start_slot)

    seen = get_seen(spec)
    signed_bls_to_execution_change = get_signed_bls_to_execution_change(
        spec, state, validator_index=validator_index, withdrawal_pubkey=pubkeys[validator_index]
    )
    yield get_filename(signed_bls_to_execution_change), signed_bls_to_execution_change

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        signed_bls_to_execution_change=signed_bls_to_execution_change,
        current_time_ms=current_time_ms,
    )
    assert result == "valid"
    assert reason is None

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(signed_bls_to_execution_change),
                "expected": "valid",
            }
        ],
    )
