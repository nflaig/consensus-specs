from types import SimpleNamespace

from eth_consensus_specs.test.context import (
    single_phase,
    spec_test,
    with_gloas_and_later,
)


def _is_bid_parent_candidate(
    spec,
    blocks,
    head_root,
    parent_root,
    *,
    head_weak=True,
    parent_strong=True,
):
    get_head = spec.get_head
    is_head_weak = spec.is_head_weak
    is_parent_strong = spec.is_parent_strong
    spec.get_head = lambda _store: spec.ForkChoiceNode(
        root=head_root,
        payload_status=spec.PAYLOAD_STATUS_EMPTY,
    )
    spec.is_head_weak = lambda _store, _root: head_weak
    spec.is_parent_strong = lambda _store, _root: parent_strong

    try:
        store = SimpleNamespace(blocks=blocks)
        bid = spec.ExecutionPayloadBid(parent_block_root=parent_root)
        return spec.is_bid_parent_candidate(store, bid)
    finally:
        spec.get_head = get_head
        spec.is_head_weak = is_head_weak
        spec.is_parent_strong = is_parent_strong


@with_gloas_and_later
@spec_test
@single_phase
def test_head_is_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    blocks = {head_root: spec.BeaconBlock(slot=spec.Slot(10))}

    assert _is_bid_parent_candidate(spec, blocks, head_root, head_root)


@with_gloas_and_later
@spec_test
@single_phase
def test_competing_head_is_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    competing_root = spec.Root(b"\x22" * 32)
    blocks = {
        head_root: spec.BeaconBlock(slot=spec.Slot(10)),
        competing_root: spec.BeaconBlock(slot=spec.Slot(10)),
    }

    assert _is_bid_parent_candidate(spec, blocks, head_root, competing_root)


@with_gloas_and_later
@spec_test
@single_phase
def test_one_slot_reorg_parent_is_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    parent_root = spec.Root(b"\x22" * 32)
    blocks = {
        head_root: spec.BeaconBlock(slot=spec.Slot(10), parent_root=parent_root),
        parent_root: spec.BeaconBlock(slot=spec.Slot(9)),
    }

    assert _is_bid_parent_candidate(spec, blocks, head_root, parent_root)


@with_gloas_and_later
@spec_test
@single_phase
def test_direct_parent_of_strong_head_is_not_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    parent_root = spec.Root(b"\x22" * 32)
    blocks = {
        head_root: spec.BeaconBlock(slot=spec.Slot(10), parent_root=parent_root),
        parent_root: spec.BeaconBlock(slot=spec.Slot(9)),
    }

    assert not _is_bid_parent_candidate(
        spec,
        blocks,
        head_root,
        parent_root,
        head_weak=False,
    )


@with_gloas_and_later
@spec_test
@single_phase
def test_direct_parent_without_strong_support_is_not_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    parent_root = spec.Root(b"\x22" * 32)
    blocks = {
        head_root: spec.BeaconBlock(slot=spec.Slot(10), parent_root=parent_root),
        parent_root: spec.BeaconBlock(slot=spec.Slot(9)),
    }

    assert not _is_bid_parent_candidate(
        spec,
        blocks,
        head_root,
        parent_root,
        parent_strong=False,
    )


@with_gloas_and_later
@spec_test
@single_phase
def test_direct_parent_separated_by_missed_slots_is_not_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    parent_root = spec.Root(b"\x22" * 32)
    blocks = {
        head_root: spec.BeaconBlock(slot=spec.Slot(10), parent_root=parent_root),
        parent_root: spec.BeaconBlock(slot=spec.Slot(5)),
    }

    assert not _is_bid_parent_candidate(spec, blocks, head_root, parent_root)


@with_gloas_and_later
@spec_test
@single_phase
def test_older_non_parent_is_not_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    old_root = spec.Root(b"\x22" * 32)
    blocks = {
        head_root: spec.BeaconBlock(slot=spec.Slot(10)),
        old_root: spec.BeaconBlock(slot=spec.Slot(8)),
    }

    assert not _is_bid_parent_candidate(spec, blocks, head_root, old_root)


@with_gloas_and_later
@spec_test
@single_phase
def test_unknown_parent_is_not_candidate(spec):
    head_root = spec.Root(b"\x11" * 32)
    unknown_root = spec.Root(b"\x22" * 32)
    blocks = {head_root: spec.BeaconBlock(slot=spec.Slot(10))}

    assert not _is_bid_parent_candidate(spec, blocks, head_root, unknown_root)
