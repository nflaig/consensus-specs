from random import Random

from eth_consensus_specs.test.context import (
    spec_state_test,
    with_gloas_and_later,
)
from eth_consensus_specs.test.helpers.proposer_slashings import (
    assert_process_proposer_slashing,
    prepare_process_proposer_slashing,
    run_proposer_slashing_processing,
)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_kept_current_epoch(spec, state):
    """
    Test that the builder pending payment is kept when the proposer is slashed in the same
    epoch as the proposal. Slashed proposers are skipped at quorum instead.

    Input State Configured:
        - state advanced by 2 epochs
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: In current epoch
        - builder_pending_payments: Contains entry for slashed proposer with amount = MIN_ACTIVATION_BALANCE

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: unchanged
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=Random(1001).randrange(spec.SLOTS_PER_EPOCH),
        parent_root_2=b"\x99" * 32,  # Make headers different
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x42" * 20,
        builder_payment_weight=1000,
    )

    # Verify the slashing is for the current epoch
    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert spec.compute_epoch_at_slot(slashed_slot) == spec.get_current_epoch(state)

    pre_state = state.copy()

    yield from run_proposer_slashing_processing(spec, state, proposer_slashing)

    assert_process_proposer_slashing(
        spec,
        state,
        pre_state,
        proposer_slashing,
    )


@with_gloas_and_later
@spec_state_test
def test_builder_payment_kept_previous_epoch(spec, state):
    """
    Test that the builder pending payment is kept when the proposer is slashed in the epoch
    after the proposal.

    Input State Configured:
        - state advanced by 2 epochs, then 1 additional epoch after slashing setup
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: In previous epoch
        - builder_pending_payments: Contains entry for slashed proposer with amount = MIN_ACTIVATION_BALANCE

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: unchanged
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        advance_epochs_after=1,  # Slot will be in previous epoch
        slot_offset=Random(1002).randrange(spec.SLOTS_PER_EPOCH),
        parent_root_2=b"\x99" * 32,  # Make headers different
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x43" * 20,
        builder_payment_weight=1000,
    )

    # Verify the slashing is for the previous epoch
    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert spec.compute_epoch_at_slot(slashed_slot) == spec.get_previous_epoch(state)

    pre_state = state.copy()

    yield from run_proposer_slashing_processing(spec, state, proposer_slashing)

    assert_process_proposer_slashing(
        spec,
        state,
        pre_state,
        proposer_slashing,
    )


@with_gloas_and_later
@spec_state_test
def test_builder_payment_kept_foreign_equivocation(spec, state):
    """
    Test that a proposer slashing of a different validator equivocating on the payment's
    slot leaves the builder pending payment intact.

    Input State Configured:
        - builder_pending_payments: current-epoch entry recorded for the payment proposer
        - proposer_slashing: valid slashing of a different validator for the same slot,
          within the 2-epoch window

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: unchanged
    """
    active = spec.get_active_validator_indices(state, spec.get_current_epoch(state))
    payment_proposer = active[0]  # the payment is recorded for this proposer
    slashed_proposer = active[-1]  # equivocates on the slot, not the payment's proposer
    assert payment_proposer != slashed_proposer

    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=Random(2024).randrange(spec.SLOTS_PER_EPOCH),
        proposer_index=slashed_proposer,
        parent_root_2=b"\x99" * 32,  # Make headers different
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x42" * 20,
        builder_payment_weight=1000,
        builder_payment_proposer_index=payment_proposer,
    )

    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert spec.compute_epoch_at_slot(slashed_slot) == spec.get_current_epoch(state)

    pre_state = state.copy()

    yield from run_proposer_slashing_processing(spec, state, proposer_slashing)

    assert_process_proposer_slashing(
        spec,
        state,
        pre_state,
        proposer_slashing,
    )
