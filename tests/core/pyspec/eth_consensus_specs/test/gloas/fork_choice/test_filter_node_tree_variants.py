from eth_consensus_specs.test.context import (
    spec_state_test,
    with_gloas_and_later,
    with_presets,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    get_valid_attestation_at_slot,
    state_transition_with_full_block,
)
from eth_consensus_specs.test.helpers.attester_slashings import (
    get_valid_attester_slashing_by_indices,
)
from eth_consensus_specs.test.helpers.block import build_empty_block_for_next_slot
from eth_consensus_specs.test.helpers.constants import MINIMAL
from eth_consensus_specs.test.helpers.execution_payload import (
    build_signed_execution_payload_envelope,
    commit_bid_to_execution_payload,
)
from eth_consensus_specs.test.helpers.fork_choice import (
    add_execution_payload,
    apply_next_slots_with_attestations,
    on_tick_and_append_step,
    output_head_check,
    setup_finalized_store,
    tick_and_add_block,
    tick_and_run_on_attestation,
    tick_store_to_slot,
)
from eth_consensus_specs.test.helpers.state import (
    next_slots,
    state_transition_and_sign_block,
)


@with_gloas_and_later
@with_presets([MINIMAL], reason="too slow")
@spec_state_test
def test_get_head_prunes_childless_unviable_full_variant(spec, state):
    """
    Reproduces issue #5496: a childless FULL payload-status variant of a block
    that fails the FFG test must not be returned by get_head.

    Block B is built on the justified checkpoint from a fork of its state, so
    its voting source is one epoch older than the store's justified checkpoint
    and B fails the FFG test once the store advances past epoch
    `justified_epoch + 1`. K builds on EMPTY(B)
    through a chain of blocks that carry attestations targeting the
    (B, justified_epoch + 1) checkpoint, so K's branch naturally justifies it
    and K's voting source is pulled up to the store's justified checkpoint,
    making EMPTY(B) viable.
    FULL(B) exists because B's envelope is delivered, but it is childless and
    never passes the FFG test, so get_head must not return it.
    """
    store, _, test_steps = yield from setup_finalized_store(spec, state)

    justified_epoch = store.justified_checkpoint.epoch
    assert store.finalized_checkpoint.epoch == justified_epoch - 1

    justified_state = store.block_states[store.justified_checkpoint.root]

    # B is the first block of epoch `justified_epoch + 1` on a fork of the
    # justified state. The fork has no on-chain votes, so B's voting source is
    # the fork's greatest justified checkpoint, one epoch older than the
    # store's justified checkpoint: `justified_epoch - 1`. B passes the FFG
    # test while the store is at epoch `justified_epoch + 1` and fails from
    # epoch `justified_epoch + 2` onwards.
    b_slot = spec.compute_start_slot_at_epoch(justified_epoch + 1)
    fork_state = justified_state.copy()
    next_slots(spec, fork_state, b_slot - fork_state.slot - 1)
    b_block = build_empty_block_for_next_slot(spec, fork_state)
    assert b_block.slot == b_slot
    commit_bid_to_execution_payload(spec, fork_state, b_block)
    signed_b = state_transition_and_sign_block(spec, fork_state, b_block)
    b_root = signed_b.message.hash_tree_root()
    yield from tick_and_add_block(spec, store, signed_b, test_steps)

    # Deliver B's envelope so the FULL(B) variant exists
    b_state = store.block_states[b_root]
    envelope = build_signed_execution_payload_envelope(spec, b_state, b_root, signed_b)
    yield from add_execution_payload(spec, store, envelope, test_steps)
    assert spec.is_payload_verified(store, b_root)

    # Tick to the start of the next epoch so attestations for B's epoch can be
    # processed
    k_slot = spec.compute_start_slot_at_epoch(justified_epoch + 2)
    on_tick_and_append_step(
        spec, store, spec.compute_time_at_slot_ms(store.genesis_time_ms, k_slot), test_steps
    )

    # Attest B's FULL variant for every slot of B's epoch after the first
    # (post-Electra, each slot has a single aggregate), so the B branch
    # outweighs the main chain in the LMD-GHOST walk
    att_state = b_state.copy()
    for slot in range(b_slot + 1, k_slot):
        next_slots(spec, att_state, slot - att_state.slot)
        attestation = get_valid_attestation_at_slot(
            att_state, spec, slot, beacon_block_root=b_root, payload_index=1
        )
        yield from tick_and_run_on_attestation(spec, store, attestation, test_steps)

    # K's branch justifies the (B, justified_epoch + 1) checkpoint by including
    # attestations targeting B in its blocks, so that K's voting source is
    # naturally pulled up to the store's justified checkpoint once K is
    # processed. Attesting all but the first and last slots of B's epoch
    # exceeds 2/3 of the total active balance, which justifies the checkpoint
    # at the next epoch boundary. (The first slot is skipped since same-slot
    # attestations must carry index 0, and the last slot is skipped since its
    # attestations could only be included after the epoch boundary.)
    att_state = b_state.copy()
    slot_attestations = []
    for slot in range(b_slot + 1, k_slot - 1):
        next_slots(spec, att_state, slot - att_state.slot)
        slot_attestations.append(
            get_valid_attestation_at_slot(
                att_state, spec, slot, beacon_block_root=b_root, payload_index=1
            )
        )

    # Intermediate blocks on the B fork carry the attestations, up to the
    # block-level maximum per block
    branch_state = b_state.copy()
    branch_root = b_root
    for att_start in range(0, len(slot_attestations), spec.MAX_ATTESTATIONS_ELECTRA):
        attestations = slot_attestations[att_start : att_start + spec.MAX_ATTESTATIONS_ELECTRA]
        # The block must be built after the latest attested slot so that the
        # attestation inclusion delay is satisfied
        branch_slot = attestations[-1].data.slot + 1
        next_slots(spec, branch_state, branch_slot - branch_state.slot - 1)
        branch_block = build_empty_block_for_next_slot(spec, branch_state)
        assert branch_block.slot == branch_slot
        for attestation in attestations:
            branch_block.body.attestations.append(attestation)
        signed_branch_block = state_transition_and_sign_block(spec, branch_state, branch_block)
        yield from tick_and_add_block(spec, store, signed_branch_block, test_steps)
        branch_root = signed_branch_block.message.hash_tree_root()

    # K builds on the last branch block at the start of the next epoch
    k_state = store.block_states[branch_root].copy()
    next_slots(spec, k_state, k_slot - k_state.slot - 1)
    k_block = build_empty_block_for_next_slot(spec, k_state)
    assert k_block.slot == k_slot
    signed_k = state_transition_and_sign_block(spec, k_state, k_block)
    k_root = signed_k.message.hash_tree_root()
    yield from tick_and_add_block(spec, store, signed_k, test_steps)
    # K must claim an EMPTY parent for the B branch to stay viable
    assert spec.get_parent_payload_status(store, signed_k.message) == spec.PAYLOAD_STATUS_EMPTY
    # K's branch justifies the (B, justified_epoch + 1) checkpoint, so K's
    # voting source is pulled up to the store's justified checkpoint, which
    # advances to (B, justified_epoch + 1)
    assert store.justified_checkpoint.epoch == justified_epoch + 1
    assert store.justified_checkpoint.root == b_root
    assert store.unrealized_justifications[k_root] == store.justified_checkpoint

    # The store is already at the start of epoch `justified_epoch + 2`, where
    # K was built. With B's voting source at `justified_epoch - 1`, B is more
    # than two epochs behind the store's current epoch, so it fails the FFG
    # test right here: no further advancement is needed for B to be filtered
    # out.

    full_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_FULL)
    empty_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_EMPTY)

    # B fails the FFG test while K passes it
    assert spec.get_voting_source(store, b_root).epoch + 2 < spec.get_current_store_epoch(store)
    assert spec.get_voting_source(store, b_root).epoch != store.justified_checkpoint.epoch
    assert spec.get_voting_source(store, k_root).epoch == store.justified_checkpoint.epoch

    # The childless FULL(B) variant is not viable and must not be the head
    head = spec.get_head(store)
    assert head != full_b_node
    assert head.root == k_root
    assert head.payload_status == spec.PAYLOAD_STATUS_EMPTY

    filtered_tree = spec.get_filtered_node_tree(store)
    assert full_b_node not in filtered_tree
    assert empty_b_node in filtered_tree

    output_head_check(spec, store, test_steps)
    yield "steps", test_steps


@with_gloas_and_later
@with_presets([MINIMAL], reason="too slow")
@spec_state_test
def test_get_head_prunes_childless_unviable_empty_variant(spec, state):
    """
    Mirror of test_get_head_prunes_childless_unviable_full_variant:
    a childless EMPTY payload-status variant of a block that fails the FFG test
    must not be returned by get_head or included in get_filtered_node_tree.

    Block B is built on the justified checkpoint from a fork of its state, so
    its voting source is one epoch older than the store's justified checkpoint
    and B fails the FFG test once the store advances past epoch
    `justified_epoch + 1`. K builds on FULL(B) through a chain of blocks that
    carry attestations targeting the (B, justified_epoch + 1) checkpoint, so K's
    branch naturally justifies it and K's voting source is pulled up to the
    store's justified checkpoint, making FULL(B) viable.
    EMPTY(B) exists because B was processed, but it is childless (since K built
    on FULL(B)) and never passes the FFG test, so get_head must not return it
    and get_filtered_node_tree must prune it.
    """
    store, _, test_steps = yield from setup_finalized_store(spec, state)

    justified_epoch = store.justified_checkpoint.epoch
    assert store.finalized_checkpoint.epoch == justified_epoch - 1

    justified_state = store.block_states[store.justified_checkpoint.root]

    # B is the first block of epoch `justified_epoch + 1` on a fork of the
    # justified state. The fork has no on-chain votes, so B's voting source is
    # the fork's greatest justified checkpoint, one epoch older than the
    # store's justified checkpoint: `justified_epoch - 1`. B passes the FFG
    # test while the store is at epoch `justified_epoch + 1` and fails from
    # epoch `justified_epoch + 2` onwards.
    b_slot = spec.compute_start_slot_at_epoch(justified_epoch + 1)
    fork_state = justified_state.copy()
    next_slots(spec, fork_state, b_slot - fork_state.slot - 1)
    b_block = build_empty_block_for_next_slot(spec, fork_state)
    assert b_block.slot == b_slot
    commit_bid_to_execution_payload(spec, fork_state, b_block)
    signed_b = state_transition_and_sign_block(spec, fork_state, b_block)
    b_root = signed_b.message.hash_tree_root()
    yield from tick_and_add_block(spec, store, signed_b, test_steps)

    # Deliver B's envelope so the FULL(B) variant exists
    b_state = store.block_states[b_root]
    envelope = build_signed_execution_payload_envelope(spec, b_state, b_root, signed_b)
    yield from add_execution_payload(spec, store, envelope, test_steps)
    assert spec.is_payload_verified(store, b_root)

    # Tick to the start of the next epoch so attestations for B's epoch can be
    # processed
    k_slot = spec.compute_start_slot_at_epoch(justified_epoch + 2)
    on_tick_and_append_step(
        spec, store, spec.compute_time_at_slot_ms(store.genesis_time_ms, k_slot), test_steps
    )

    # Attest B's FULL variant for every slot of B's epoch after the first
    # (post-Electra, each slot has a single aggregate), so the B branch
    # outweighs the main chain in the LMD-GHOST walk
    att_state = b_state.copy()
    for slot in range(b_slot + 1, k_slot):
        next_slots(spec, att_state, slot - att_state.slot)
        attestation = get_valid_attestation_at_slot(
            att_state, spec, slot, beacon_block_root=b_root, payload_index=1
        )
        yield from tick_and_run_on_attestation(spec, store, attestation, test_steps)

    # K's branch justifies the (B, justified_epoch + 1) checkpoint by including
    # attestations targeting B in its blocks, so that K's voting source is
    # naturally pulled up to the store's justified checkpoint once K is
    # processed.
    att_state = b_state.copy()
    slot_attestations = []
    for slot in range(b_slot + 1, k_slot - 1):
        next_slots(spec, att_state, slot - att_state.slot)
        slot_attestations.append(
            get_valid_attestation_at_slot(
                att_state, spec, slot, beacon_block_root=b_root, payload_index=1
            )
        )

    # Intermediate blocks on the B fork carry the attestations.
    # The first block building directly on B explicitly claims FULL parent payload status
    # by matching parent_block_hash to signed_b's bid block_hash.
    branch_state = b_state.copy()
    branch_root = b_root
    first_branch_block = None
    for att_start in range(0, len(slot_attestations), spec.MAX_ATTESTATIONS_ELECTRA):
        attestations = slot_attestations[att_start : att_start + spec.MAX_ATTESTATIONS_ELECTRA]
        branch_slot = attestations[-1].data.slot + 1
        next_slots(spec, branch_state, branch_slot - branch_state.slot - 1)
        branch_block = build_empty_block_for_next_slot(spec, branch_state)
        assert branch_block.slot == branch_slot
        for attestation in attestations:
            branch_block.body.attestations.append(attestation)
        if branch_root == b_root:
            branch_block.body.signed_execution_payload_bid.message.parent_block_hash = (
                signed_b.message.body.signed_execution_payload_bid.message.block_hash
            )
            branch_block.body.signed_execution_payload_bid.message.block_hash = spec.Hash32(
                b"\x02" + b"\x00" * 31
            )
        signed_branch_block = state_transition_and_sign_block(spec, branch_state, branch_block)
        if first_branch_block is None:
            first_branch_block = signed_branch_block.message
        yield from tick_and_add_block(spec, store, signed_branch_block, test_steps)
        branch_root = signed_branch_block.message.hash_tree_root()

    # Confirm the branch directly builds on FULL(B)
    assert spec.get_parent_payload_status(store, first_branch_block) == spec.PAYLOAD_STATUS_FULL

    # K builds on the last branch block at the start of the next epoch
    k_state = store.block_states[branch_root].copy()
    next_slots(spec, k_state, k_slot - k_state.slot - 1)
    k_block = build_empty_block_for_next_slot(spec, k_state)
    k_block.body.signed_execution_payload_bid.message.block_hash = spec.Hash32(
        b"\x03" + b"\x00" * 31
    )
    assert k_block.slot == k_slot
    signed_k = state_transition_and_sign_block(spec, k_state, k_block)
    k_root = signed_k.message.hash_tree_root()
    yield from tick_and_add_block(spec, store, signed_k, test_steps)

    # K's branch justifies the (B, justified_epoch + 1) checkpoint, so K's
    # voting source is pulled up to the store's justified checkpoint, which
    # advances to (B, justified_epoch + 1)
    assert store.justified_checkpoint.epoch == justified_epoch + 1
    assert store.justified_checkpoint.root == b_root
    assert store.unrealized_justifications[k_root] == store.justified_checkpoint

    full_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_FULL)
    empty_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_EMPTY)

    # B fails the FFG test while K passes it
    assert spec.get_voting_source(store, b_root).epoch + 2 < spec.get_current_store_epoch(store)
    assert spec.get_voting_source(store, b_root).epoch != store.justified_checkpoint.epoch
    assert spec.get_voting_source(store, k_root).epoch == store.justified_checkpoint.epoch

    # The childless EMPTY(B) variant is not viable and must not be the head
    head = spec.get_head(store)
    assert head != empty_b_node
    assert head.root == k_root

    filtered_tree = spec.get_filtered_node_tree(store)
    assert empty_b_node not in filtered_tree
    assert full_b_node in filtered_tree

    output_head_check(spec, store, test_steps)
    yield "steps", test_steps


@with_gloas_and_later
@with_presets([MINIMAL], reason="too slow")
@spec_state_test
def test_get_filtered_node_tree_contains_both_viable_childless_variants(spec, state):
    """
    Verifies that when both FULL(B) and EMPTY(B) variants of a block pass the FFG test
    and are both childless, get_filtered_node_tree contains both variants.
    """
    store, state, test_steps = yield from setup_finalized_store(spec, state)

    # Build block B directly on the justified state (passes FFG test)
    b_block = build_empty_block_for_next_slot(spec, state)
    commit_bid_to_execution_payload(spec, state, b_block)
    signed_b = state_transition_and_sign_block(spec, state, b_block)
    b_root = signed_b.message.hash_tree_root()
    yield from tick_and_add_block(spec, store, signed_b, test_steps)

    # Deliver B's envelope so both FULL(B) and EMPTY(B) variants exist in store
    b_state = store.block_states[b_root]
    envelope = build_signed_execution_payload_envelope(spec, b_state, b_root, signed_b)
    yield from add_execution_payload(spec, store, envelope, test_steps)
    assert spec.is_payload_verified(store, b_root)

    full_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_FULL)
    empty_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_EMPTY)

    # Both childless variants pass the FFG test and must be present in get_filtered_node_tree
    filtered_tree = spec.get_filtered_node_tree(store)
    assert full_b_node in filtered_tree
    assert empty_b_node in filtered_tree

    output_head_check(spec, store, test_steps)
    yield "steps", test_steps


@with_gloas_and_later
@with_presets([MINIMAL], reason="too slow")
@spec_state_test
def test_get_head_returns_viable_full_variant_with_unviable_child(spec, state):
    """
    The FULL payload-status variant of a block that passes the FFG test must be
    returned by get_head when its only child fails the FFG test.

    The votes for epoch 4 stay just below 2/3 of the total active balance until
    block B, the first block of epoch 5, includes one more vote. B's voting
    source is therefore pulled up to epoch 4. S builds on FULL(B) and slashes
    that attester, so S's voting source stays at epoch 3. Once the store's
    justified checkpoint advances to epoch 4 at the start of epoch 6, S fails
    the FFG test while B passes it. EMPTY(B) is childless and FULL(B) is
    supported by the votes for S, so get_head must return FULL(B).
    """
    store, state, test_steps = yield from setup_finalized_store(spec, state)
    assert spec.compute_epoch_at_slot(spec.get_current_slot(store)) == 4
    assert store.justified_checkpoint.epoch == 3

    # The balance of `attesters` is just below 2/3 of the total active balance,
    # the vote of `late_attester` brings it to 2/3
    total_balance = spec.get_total_active_balance(state)
    attesters = set()
    attesting_balance = 0
    late_attester = None
    for index in spec.get_active_validator_indices(state, spec.get_current_epoch(state)):
        attesting_balance += state.validators[index].effective_balance
        if attesting_balance * 3 >= total_balance * 2:
            late_attester = index
            break
        attesters.add(index)
    assert late_attester is not None

    def participation_fn(_slot, _committee_index, committee):
        return committee & attesters

    # Fill epoch 4 with the votes of `attesters`, epoch 4 is not justified
    state, store, _ = yield from apply_next_slots_with_attestations(
        spec,
        state,
        store,
        spec.SLOTS_PER_EPOCH - 1,
        fill_cur_epoch=True,
        fill_prev_epoch=False,
        test_steps=test_steps,
        participation_fn=participation_fn,
    )
    assert store.unrealized_justified_checkpoint.epoch == 3

    # B is the first block of epoch 5 and includes the vote of `late_attester`
    # for epoch 4, so its voting source is pulled up to epoch 4. B commits to a
    # distinct block hash so that a child can build on its payload.
    _, committee_index, slot = spec.get_committee_assignment(
        state, spec.get_current_epoch(state), late_attester
    )
    late_attestation = get_valid_attestation(
        spec,
        state,
        slot=slot,
        index=committee_index,
        filter_participant_set=lambda committee: committee & {late_attester},
        signed=True,
    )
    b_block = build_empty_block_for_next_slot(spec, state)
    b_block.body.attestations.append(late_attestation)
    b_block_hash = spec.Hash32(b"\x42" * 32)
    b_block.body.signed_execution_payload_bid.message.block_hash = b_block_hash
    signed_b = state_transition_with_full_block(
        spec,
        state,
        fill_cur_epoch=True,
        fill_prev_epoch=False,
        participation_fn=participation_fn,
        block=b_block,
    )
    yield from tick_and_add_block(spec, store, signed_b, test_steps)
    b_root = signed_b.message.hash_tree_root()
    assert store.unrealized_justifications[b_root].epoch == 4

    # Deliver B's envelope so the FULL(B) variant exists
    envelope = build_signed_execution_payload_envelope(
        spec, store.block_states[b_root], b_root, signed_b
    )
    yield from add_execution_payload(spec, store, envelope, test_steps)
    assert spec.is_payload_verified(store, b_root)

    # S builds on FULL(B) and slashes `late_attester`, so its voting source
    # stays at epoch 3
    attester_slashing = get_valid_attester_slashing_by_indices(
        spec, state, [late_attester], signed_1=True, signed_2=True
    )
    s_block = build_empty_block_for_next_slot(spec, state)
    s_block.body.attester_slashings.append(attester_slashing)
    s_block.body.signed_execution_payload_bid.message.parent_block_hash = b_block_hash
    signed_s = state_transition_and_sign_block(spec, state, s_block)
    yield from tick_and_add_block(spec, store, signed_s, test_steps)
    s_root = signed_s.message.hash_tree_root()
    assert spec.get_parent_payload_status(store, signed_s.message) == spec.PAYLOAD_STATUS_FULL
    assert store.unrealized_justifications[s_root].epoch == 3
    assert spec.get_head(store).root == s_root

    # Attest S, the votes support FULL(B)
    attestation = get_valid_attestation_at_slot(state, spec, state.slot)
    yield from tick_and_run_on_attestation(spec, store, attestation, test_steps)

    # At the start of epoch 6, the store's justified checkpoint advances to
    # epoch 4. S fails the FFG test, B passes it.
    tick_store_to_slot(spec, store, spec.compute_start_slot_at_epoch(6), test_steps)
    assert store.justified_checkpoint.epoch == 4
    assert spec.get_voting_source(store, b_root).epoch == 4
    assert spec.get_voting_source(store, s_root).epoch == 3

    full_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_FULL)
    empty_b_node = spec.ForkChoiceNode(root=b_root, payload_status=spec.PAYLOAD_STATUS_EMPTY)
    pending_s_node = spec.ForkChoiceNode(root=s_root, payload_status=spec.PAYLOAD_STATUS_PENDING)

    # Both variants of B are in the filtered tree, S is not
    filtered_tree = spec.get_filtered_node_tree(store)
    assert full_b_node in filtered_tree
    assert empty_b_node in filtered_tree
    assert pending_s_node not in filtered_tree

    # FULL(B) carries the votes for S and is the head
    assert spec.get_weight(store, full_b_node) > spec.get_weight(store, empty_b_node)
    assert spec.get_head(store) == full_b_node

    output_head_check(spec, store, test_steps)
    yield "steps", test_steps
