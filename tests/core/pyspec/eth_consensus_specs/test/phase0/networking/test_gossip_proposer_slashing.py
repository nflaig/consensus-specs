from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_all_phases,
)
from eth_consensus_specs.test.helpers.gossip import (
    get_filename,
    get_seen,
    get_store_from_state,
    run_validate_gossip,
)
from eth_consensus_specs.test.helpers.proposer_slashings import (
    get_valid_proposer_slashing,
)


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__valid(spec, state):
    """
    Test that a valid proposer slashing passes gossip validation.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
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
                "message": get_filename(proposer_slashing),
                "expected": "valid",
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__ignore_already_seen(spec, state):
    """
    Test that a duplicate proposer slashing is ignored.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    messages = []
    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)

    yield get_filename(proposer_slashing), proposer_slashing

    # First validation should pass
    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "valid"
    assert reason is None
    messages.append(
        {
            "current_time_ms": int(current_time_ms),
            "message": get_filename(proposer_slashing),
            "expected": "valid",
        }
    )

    # Second validation should be ignored
    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "ignore"
    assert reason == "already seen proposer slashing for this proposer"
    messages.append(
        {
            "current_time_ms": int(current_time_ms),
            "message": get_filename(proposer_slashing),
            "expected": "ignore",
            "reason": reason,
        }
    )

    yield "messages", "meta", messages


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__reject_slots_not_matching(spec, state):
    """
    Test that a proposer slashing with mismatched header slots is rejected.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)

    # Modify second header slot to not match
    proposer_slashing.signed_header_2.message.slot = (
        proposer_slashing.signed_header_1.message.slot + 1
    )

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "header slots do not match"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__reject_proposer_indices_not_matching(spec, state):
    """
    Test that a proposer slashing with mismatched proposer indices is rejected.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)

    # Modify second header proposer_index to not match
    proposer_slashing.signed_header_2.message.proposer_index = (
        proposer_slashing.signed_header_1.message.proposer_index + 1
    )

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "header proposer indices do not match"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__reject_headers_identical(spec, state):
    """
    Test that a proposer slashing with identical headers is rejected.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)

    # Make headers identical by copying header_1 to header_2
    proposer_slashing.signed_header_2 = proposer_slashing.signed_header_1.copy()

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "headers are not different"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__reject_proposer_index_out_of_range(spec, state):
    """
    Test that a proposer slashing with proposer index out of range is rejected.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)

    # Set proposer index to be out of range
    invalid_index = len(state.validators) + 100
    proposer_slashing.signed_header_1.message.proposer_index = invalid_index
    proposer_slashing.signed_header_2.message.proposer_index = invalid_index

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "proposer index out of range"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__reject_proposer_not_slashable(spec, state):
    """
    Test that a proposer slashing for a non-slashable proposer is rejected.
    """
    yield "topic", "meta", "proposer_slashing"

    seen = get_seen(spec)

    # Create a valid proposer slashing
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)
    proposer_index = proposer_slashing.signed_header_1.message.proposer_index

    # Make the proposer not slashable by setting slashed=True
    state.validators[proposer_index].slashed = True

    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)
    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "proposer is not slashable"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__reject_invalid_signature_1(spec, state):
    """
    Test that a proposer slashing with invalid first signature is rejected.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a proposer slashing with only second header signed
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=False, signed_2=True)

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "invalid proposer slashing signature"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
@always_bls
def test_gossip_proposer_slashing__reject_invalid_signature_2(spec, state):
    """
    Test that a proposer slashing with invalid second signature is rejected.
    """
    yield "topic", "meta", "proposer_slashing"
    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, state.slot)

    seen = get_seen(spec)

    # Create a proposer slashing with only first header signed
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=False)

    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "invalid proposer slashing signature"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__valid_activation_epoch_at_clock_epoch(spec, state):
    """
    Test that a proposer slashing is validated against the head state advanced to
    the current slot: a proposer whose activation epoch is the clock epoch is
    slashable, although the head block is in the previous epoch.
    """
    yield "topic", "meta", "proposer_slashing"

    seen = get_seen(spec)

    clock_epoch = spec.get_current_epoch(state) + 1
    current_slot = spec.compute_start_slot_at_epoch(clock_epoch)
    proposer_slashing = get_valid_proposer_slashing(
        spec, state, slot=current_slot, signed_1=True, signed_2=True
    )
    proposer_index = proposer_slashing.signed_header_1.message.proposer_index
    state.validators[proposer_index].activation_epoch = clock_epoch

    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, current_slot)
    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
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
                "message": get_filename(proposer_slashing),
                "expected": "valid",
            }
        ],
    )


@with_all_phases
@spec_state_test
def test_gossip_proposer_slashing__reject_withdrawable_epoch_at_clock_epoch(spec, state):
    """
    Test that a proposer slashing is validated against the head state advanced to
    the current slot: a proposer whose withdrawable epoch is the clock epoch is not
    slashable, although the head block is in the previous epoch.
    """
    yield "topic", "meta", "proposer_slashing"

    seen = get_seen(spec)

    clock_epoch = spec.get_current_epoch(state) + 1
    current_slot = spec.compute_start_slot_at_epoch(clock_epoch)
    proposer_slashing = get_valid_proposer_slashing(spec, state, signed_1=True, signed_2=True)
    proposer_index = proposer_slashing.signed_header_1.message.proposer_index
    state.validators[proposer_index].exit_epoch = clock_epoch
    state.validators[proposer_index].withdrawable_epoch = clock_epoch

    yield "state", state

    store, signed_anchor = get_store_from_state(spec, state)
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]
    current_time_ms = spec.compute_time_at_slot_ms(store.genesis_time_ms, current_slot)
    yield get_filename(proposer_slashing), proposer_slashing

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        proposer_slashing=proposer_slashing,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "proposer is not slashable"

    yield (
        "messages",
        "meta",
        [
            {
                "current_time_ms": int(current_time_ms),
                "message": get_filename(proposer_slashing),
                "expected": "reject",
                "reason": reason,
            }
        ],
    )
