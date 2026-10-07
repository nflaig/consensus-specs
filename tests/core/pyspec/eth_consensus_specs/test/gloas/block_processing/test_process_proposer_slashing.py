from random import Random

from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_gloas_and_later,
)
from eth_consensus_specs.test.helpers.attester_slashings import (
    get_valid_attester_slashing_by_indices,
)
from eth_consensus_specs.test.helpers.proposer_slashings import (
    assert_process_proposer_slashing,
    get_valid_proposer_slashing,
    prepare_process_proposer_slashing,
    run_proposer_slashing_processing,
)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_deletion_current_epoch(spec, state):
    """
    Test that builder pending payment is deleted when proposer is slashed in the same epoch as the proposal.

    Input State Configured:
        - state advanced by 2 epochs
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: In current epoch
        - builder_pending_payments: Contains entry for slashed proposer with amount = MIN_ACTIVATION_BALANCE

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Entry for slashed proposer removed (slashing within 2-epoch window)
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
def test_builder_payment_deletion_previous_epoch(spec, state):
    """
    Test that builder pending payment is deleted when proposer is slashed in the epoch after the proposal.

    Input State Configured:
        - state advanced by 2 epochs, then 1 additional epoch after slashing setup
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: In previous epoch
        - builder_pending_payments: Contains entry for slashed proposer with amount = MIN_ACTIVATION_BALANCE

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Entry for slashed proposer removed (slashing within 2-epoch window)
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
def test_builder_payment_deletion_too_late(spec, state):
    """
    Test that builder pending payment is NOT deleted when slashing comes more than two epochs after the proposal slot.

    Input State Configured:
        - state advanced by 2 epochs, then 2 additional epochs after slashing setup
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: More than 2 epochs before current epoch

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Unchanged (slashing outside 2-epoch window, no payment deletion)
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        advance_epochs_after=2,  # Slot will be outside 2-epoch window
        slot_offset=Random(1003).randrange(spec.SLOTS_PER_EPOCH),
        parent_root_2=b"\x99" * 32,  # Make headers different
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x46" * 20,
        builder_payment_weight=1000,
    )

    pre_state = state.copy()

    yield from run_proposer_slashing_processing(spec, state, proposer_slashing)

    # Verify standard slashing effects and that payments are unmodified
    assert_process_proposer_slashing(
        spec,
        state,
        pre_state,
        proposer_slashing,
    )


@with_gloas_and_later
@spec_state_test
def test_builder_payment_empty_current_epoch(spec, state):
    """
    Test slashing succeeds when no builder payment exists at current epoch slot.

    Input State Configured:
        - state advanced by 2 epochs
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: In current epoch
        - builder_pending_payments: No entry set (empty slot)

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Remains empty (no change)
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=Random(1004).randrange(spec.SLOTS_PER_EPOCH),
        parent_root_2=b"\x99" * 32,
        # No builder_payment_amount - payment slot stays empty
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


@with_gloas_and_later
@spec_state_test
def test_builder_payment_empty_previous_epoch(spec, state):
    """
    Test slashing succeeds when no builder payment exists at previous epoch slot.

    Input State Configured:
        - state advanced by 2 epochs, then 1 additional epoch after slashing setup
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: In previous epoch
        - builder_pending_payments: No entry set (empty slot)

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Remains empty (no change)
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        advance_epochs_after=1,
        slot_offset=Random(1005).randrange(spec.SLOTS_PER_EPOCH),
        parent_root_2=b"\x99" * 32,
        # No builder_payment_amount - payment slot stays empty
    )

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
def test_builder_payment_empty_old_epoch(spec, state):
    """
    Test slashing succeeds when no builder payment exists at older epoch slot (outside 2-epoch window).

    Input State Configured:
        - state advanced by 2 epochs, then 2 additional epochs after slashing setup
        - proposer_slashing: Valid slashing with different parent_root values
        - proposer_slashing.signed_header_1.message.slot: More than 2 epochs before current epoch
        - builder_pending_payments: No entry set (empty slot)

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Unchanged (slot outside 2-epoch window)
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        advance_epochs_after=2,
        slot_offset=Random(1006).randrange(spec.SLOTS_PER_EPOCH),
        parent_root_2=b"\x99" * 32,
        # No builder_payment_amount - payment slot stays empty
    )

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
def test_builder_payment_deletion_current_epoch_first_slot(spec, state):
    """
    Test builder payment deletion at first slot of current epoch.

    Input State Configured:
        - state advanced to first slot of epoch
        - proposer_slashing with headers at slot 0 of current epoch
        - builder_pending_payments: Entry at payment_index = SLOTS_PER_EPOCH

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Entry at index SLOTS_PER_EPOCH deleted
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=0,  # First slot of epoch
        parent_root_2=b"\x99" * 32,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x42" * 20,
        builder_payment_weight=1000,
    )

    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert slashed_slot % spec.SLOTS_PER_EPOCH == 0

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
def test_builder_payment_deletion_current_epoch_last_slot(spec, state):
    """
    Test builder payment deletion at last slot of current epoch.

    Input State Configured:
        - state advanced to last slot of epoch
        - proposer_slashing with headers at last slot of current epoch
        - builder_pending_payments: Entry at payment_index = 2*SLOTS_PER_EPOCH - 1

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Entry at index 2*SLOTS_PER_EPOCH - 1 deleted
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=spec.SLOTS_PER_EPOCH - 1,  # Last slot of epoch
        parent_root_2=b"\x99" * 32,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x43" * 20,
        builder_payment_weight=1000,
    )

    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert slashed_slot % spec.SLOTS_PER_EPOCH == spec.SLOTS_PER_EPOCH - 1

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
def test_builder_payment_deletion_previous_epoch_first_slot(spec, state):
    """
    Test builder payment deletion at first slot of previous epoch.

    Input State Configured:
        - Headers created at first slot of an epoch
        - state advanced by 1 epoch (slot now in previous epoch)
        - builder_pending_payments: Entry at payment_index = 0

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Entry at index 0 deleted
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        advance_epochs_after=1,
        slot_offset=0,  # First slot of epoch
        parent_root_2=b"\x99" * 32,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x44" * 20,
        builder_payment_weight=1000,
    )

    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert slashed_slot % spec.SLOTS_PER_EPOCH == 0
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
def test_builder_payment_deletion_previous_epoch_last_slot(spec, state):
    """
    Test builder payment deletion at last slot of previous epoch.

    Input State Configured:
        - Headers created at last slot of an epoch
        - state advanced by 1 epoch (slot now in previous epoch)
        - builder_pending_payments: Entry at payment_index = SLOTS_PER_EPOCH - 1

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: Entry at index SLOTS_PER_EPOCH - 1 deleted
    """
    proposer_slashing, _ = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        advance_epochs_after=1,
        slot_offset=spec.SLOTS_PER_EPOCH - 1,  # Last slot of epoch
        parent_root_2=b"\x99" * 32,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        builder_payment_fee_recipient=b"\x45" * 20,
        builder_payment_weight=1000,
    )

    slashed_slot = proposer_slashing.signed_header_1.message.slot
    assert slashed_slot % spec.SLOTS_PER_EPOCH == spec.SLOTS_PER_EPOCH - 1
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
def test_builder_payment_not_deleted_foreign_equivocation(spec, state):
    """
    Test that a proposer slashing does NOT delete a builder pending payment recorded for
    a different proposer. This guards against a griefing vector: the clear is keyed only
    by header slot, so without the proposer check any validator equivocating on a slot
    could clear an honest proposer's payment for that slot.

    Input State Configured:
        - builder_pending_payments: current-epoch entry recorded for the payment proposer
        - proposer_slashing: valid slashing of a different validator for the same slot,
          within the 2-epoch window

    Output State Verified:
        - validators[slashed_index].slashed: True
        - builder_pending_payments: entry left intact (slashed validator is not the
          payment's proposer)
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


def _prepare_pending_payment(spec, state, previous_epoch=False, **kwargs):
    proposer_slashing, proposer_index = prepare_process_proposer_slashing(
        spec,
        state,
        advance_epochs=2,
        slot_offset=-spec.SLOTS_PER_EPOCH if previous_epoch else 0,
        parent_root_2=b"\x99" * 32,
        builder_payment_amount=spec.MIN_ACTIVATION_BALANCE,
        **kwargs,
    )
    slot = proposer_slashing.signed_header_1.message.slot
    payment_index = slot % spec.SLOTS_PER_EPOCH
    if not previous_epoch:
        payment_index += spec.SLOTS_PER_EPOCH
    assert state.builder_pending_payments[payment_index].withdrawal.amount > 0
    return proposer_slashing, proposer_index, payment_index


def _run_cancellation_after_slashing(spec, state, previous_epoch=False, attester=True):
    proposer_slashing, proposer_index, payment_index = _prepare_pending_payment(
        spec, state, previous_epoch=previous_epoch
    )
    if attester:
        attester_slashing = get_valid_attester_slashing_by_indices(
            spec, state, [proposer_index], signed_1=True, signed_2=True
        )
        spec.process_attester_slashing(state, attester_slashing)
    else:
        earlier_slashing = get_valid_proposer_slashing(
            spec,
            state,
            slashed_index=proposer_index,
            slot=state.slot - 1,
            signed_1=True,
            signed_2=True,
        )
        spec.process_proposer_slashing(state, earlier_slashing)

    assert state.validators[proposer_index].slashed
    assert state.builder_pending_payments[payment_index].withdrawal.amount > 0
    expected_state = state.copy()
    expected_state.builder_pending_payments[payment_index] = spec.BuilderPendingPayment.empty()

    yield from run_proposer_slashing_processing(spec, state, proposer_slashing)

    # Cancellation must not apply another penalty, reward, or exit update.
    assert state == expected_state


@with_gloas_and_later
@spec_state_test
@always_bls
def test_builder_payment_cancellation_after_attester_slashing_current_epoch(spec, state):
    yield from _run_cancellation_after_slashing(spec, state)


@with_gloas_and_later
@spec_state_test
@always_bls
def test_builder_payment_cancellation_after_attester_slashing_previous_epoch(spec, state):
    yield from _run_cancellation_after_slashing(spec, state, previous_epoch=True)


@with_gloas_and_later
@spec_state_test
@always_bls
def test_builder_payment_cancellation_after_proposer_slashing_current_epoch(spec, state):
    yield from _run_cancellation_after_slashing(spec, state, attester=False)


@with_gloas_and_later
@spec_state_test
@always_bls
def test_builder_payment_cancellation_after_proposer_slashing_previous_epoch(spec, state):
    yield from _run_cancellation_after_slashing(spec, state, previous_epoch=True, attester=False)


def _run_invalid_cancellation(spec, state, proposer_slashing):
    pre_state = state.copy()
    yield from run_proposer_slashing_processing(spec, state, proposer_slashing, valid=False)
    assert state == pre_state


@with_gloas_and_later
@spec_state_test
@always_bls
def test_builder_payment_cancellation_invalid_signature_1(spec, state):
    proposer_slashing, _, _ = _prepare_pending_payment(
        spec, state, proposer_slashed=True, signed_1=False
    )
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
@always_bls
def test_builder_payment_cancellation_invalid_signature_2(spec, state):
    proposer_slashing, _, _ = _prepare_pending_payment(
        spec, state, proposer_slashed=True, signed_2=False
    )
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_foreign_proposer(spec, state):
    proposer_slashing, proposer_index, payment_index = _prepare_pending_payment(
        spec, state, proposer_slashed=True
    )
    state.builder_pending_payments[payment_index].proposer_index = (proposer_index + 1) % len(
        state.validators
    )
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_zero_amount(spec, state):
    proposer_slashing, _, payment_index = _prepare_pending_payment(
        spec, state, proposer_index=0, proposer_slashed=True
    )
    # A non-default entry with no payment must not make a proof useful.
    state.builder_pending_payments[payment_index].withdrawal.amount = 0
    state.builder_pending_payments[payment_index].weight = 1000
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_outside_window(spec, state):
    _, proposer_index, _ = _prepare_pending_payment(spec, state, proposer_slashed=True)
    # The old slot aliases a live payment's offset, but must not clear it.
    proposer_slashing = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=proposer_index,
        slot=state.slot - 2 * spec.SLOTS_PER_EPOCH,
        signed_1=True,
        signed_2=True,
    )
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_already_settled(spec, state):
    proposer_slashing, _, payment_index = _prepare_pending_payment(
        spec, state, proposer_slashed=True
    )
    spec.settle_builder_payment(state, payment_index)
    assert len(state.builder_pending_withdrawals) == 1
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_duplicate(spec, state):
    proposer_slashing, _, _ = _prepare_pending_payment(spec, state, proposer_slashed=True)
    spec.process_proposer_slashing(state, proposer_slashing)
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_mismatched_slots(spec, state):
    proposer_slashing, _, _ = _prepare_pending_payment(spec, state, proposer_slashed=True)
    proposer_slashing.signed_header_2.message.slot += 1
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_mismatched_proposers(spec, state):
    proposer_slashing, _, _ = _prepare_pending_payment(spec, state, proposer_slashed=True)
    proposer_slashing.signed_header_2.message.proposer_index -= 1
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_identical_headers(spec, state):
    proposer_slashing, _, _ = _prepare_pending_payment(spec, state, proposer_slashed=True)
    proposer_slashing.signed_header_2 = proposer_slashing.signed_header_1.copy()
    yield from _run_invalid_cancellation(spec, state, proposer_slashing)


@with_gloas_and_later
@spec_state_test
def test_builder_payment_cancellation_prevents_quorum_settlement(spec, state):
    proposer_slashing, _, payment_index = _prepare_pending_payment(
        spec, state, previous_epoch=True, proposer_slashed=True
    )
    state.builder_pending_payments[
        payment_index
    ].weight = spec.get_builder_payment_quorum_threshold(state)
    control = state.copy()
    spec.process_builder_pending_payments(control)
    assert len(control.builder_pending_withdrawals) == 1

    yield from run_proposer_slashing_processing(spec, state, proposer_slashing)
    settled_state = state.copy()
    spec.process_builder_pending_payments(settled_state)
    assert len(settled_state.builder_pending_withdrawals) == 0
