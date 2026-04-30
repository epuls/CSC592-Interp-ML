from functools import lru_cache, partial

import pytest
import torch
import torch.utils.data
import torchvision.transforms as transforms

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.datasets.cub200 import CUB200CachedPartLabels
from protopnet.datasets.torch_extensions import ImageFolderDict, uneven_collate_fn
from protopnet.embedding import AddonLayers
from protopnet.metrics import (
    PartConsistencyScore,
    PartStabilityScore,
    PrototypeAblationScore,
    PrototypeAblationUniqueCount,
    add_gaussian_noise,
    PartQualityScore,
    PartSpecificityScore,
    PrototypeSelectivityScore,
)
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.preprocess import mean, std


@pytest.fixture()
def loaded_ppnet(num_classes=2):
    base_architecture = "vgg11"
    num_prototypes = 200
    activation = CosPrototypeActivation()  # Assume imported correctly
    num_prototypes_per_class = 1

    backbone = construct_backbone(base_architecture)
    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes,
        input_channels=backbone.latent_dimension[0],
        proto_channels=128,
    )

    ppnet = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    return ppnet


def test_interp_metrics(loaded_ppnet, seed):
    cub_meta_labels = CUB200CachedPartLabels("test/dummy_test_files/test_dataset/")
    test_dir = "test/dummy_test_files/test_dataset/images"
    img_size = 224
    normalize = transforms.Normalize(mean=mean, std=std)
    num_classes = 2
    test_batch_size = 1

    test_dataset = ImageFolderDict(
        test_dir,
        transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        cached_part_labels=cub_meta_labels,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=partial(
            uneven_collate_fn, stack_ignore_key="sample_parts_centroids"
        ),
        generator=torch.Generator().manual_seed(seed),
    )

    pss = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=False,
    )

    pss_stable = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=True,
    )

    pcs = PartConsistencyScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        part_thresh=0.8,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=True,
    )

    pas = PrototypeAblationScore()
    pau = PrototypeAblationUniqueCount()
    pqs = PartQualityScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=True,
    )

    psps = PartSpecificityScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        part_num=cub_meta_labels.get_part_num(),
        min_part_fraction=0.0,
        uncropped=True,
    )

    ps = PrototypeSelectivityScore(
        num_classes=num_classes,
        part_num=cub_meta_labels.get_part_num(),
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        uncropped=True,
    )


    intersperse_rsts_pcs = []
    intersperse_rsts_pss = []
    intersperse_rsts_pss_stable = []
    intersperse_rsts_psps = []
    intersperse_rsts_ps = []
    generator = torch.Generator()
    generator.manual_seed(seed)
    with torch.no_grad():
        for packet in test_loader:
            data = packet["img"]
            targets = packet["target"]

            sample_parts_centroids = packet["sample_parts_centroids"]
            sample_bounding_box = packet["sample_bounding_box"]

            model_outputs = loaded_ppnet(
                data, return_prototype_layer_output_dict=True
            )
            proto_acts = model_outputs["prototype_activations"]
            logits = model_outputs["logits"]
            class_connection_weights = (
                loaded_ppnet.prototype_prediction_head.class_connection_layer.weight
            )
            proto_acts_noisy = loaded_ppnet(
                add_gaussian_noise(data, generator),
                return_prototype_layer_output_dict=True,
            )["prototype_activations"]

            pcs.update(proto_acts, targets, sample_parts_centroids, sample_bounding_box)
            pas.update(proto_acts, logits, class_connection_weights)
            pau.update(proto_acts, logits, class_connection_weights)
            pqs.update(proto_acts, targets, sample_parts_centroids, sample_bounding_box)
            psps.update(proto_acts, targets, sample_parts_centroids, sample_bounding_box)
            ps.update(proto_acts, targets)
            pss.update(
                proto_acts,
                proto_acts_noisy,
                targets,
                sample_parts_centroids,
                sample_bounding_box,
            )
            pss_stable.update(
                proto_acts,
                proto_acts,
                targets,
                sample_parts_centroids,
                sample_bounding_box,
            )

            intersperse_rsts_pcs.append(pcs.compute())
            intersperse_rsts_pss.append(pss.compute())
            intersperse_rsts_pss_stable.append(pss_stable.compute())
            intersperse_rsts_psps.append(psps.compute())
            intersperse_rsts_ps.append(ps.compute())

    pss_score = pss.compute()
    pcs_score = pcs.compute()
    pas_score = pas.compute()
    pau_score = pau.compute()
    pqs_score = pqs.compute()
    psps_score = psps.compute()
    pss_stable_score = pss_stable.compute()
    ps_score = ps.compute()

    assert pcs_score == 0.0, "pcs_score test failed"
    assert pss_stable_score == 1.0, "pss_stable_score test failed"
    assert pas_score >= 0.0 and pas_score <= 1.0, "pas_score test failed"
    assert pau_score >= 0, "pau_score test failed"
    assert pqs_score >= 0.0 and pqs_score <= 1.0, "pqs_score test failed"
    assert psps_score >= 0.0 and psps_score <= 1.0, "psps_score test failed"
    assert 0.0 <= ps_score <= 1.0,  "ps_score test failed"

    assert intersperse_rsts_pcs[-1] == 0.0, "intersperse_rsts_pcs test failed"
    assert (
        intersperse_rsts_pss_stable[-1] == 1.0
    ), "intersperse_rsts_pss_stable test failed"


@pytest.mark.parametrize(
    "proto_per_class,activation_position,all_sample_parts_centroids,all_sample_bounding_box,expected_proto_to_part,expected_proto_part_mask",
    [
        pytest.param(
            3,
            (0, 0),
            [
                torch.tensor(
                    [
                        [3, 10 / 224, 10 / 224],
                        [4, 20 / 224, 20 / 224],
                        [5, 112 / 224, 112 / 224],
                        [6, 90 / 224, 90 / 224],
                    ]
                )
            ],
            torch.tensor([[10, 10, 224, 224]]) / 224,
            [
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
            [
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
        ),
        pytest.param(
            2,
            (7, 7),
            [
                torch.tensor(
                    [
                        [3, 10 / 224, 10 / 224],
                        [4, 20 / 224, 20 / 224],
                        [5, 112 / 224, 112 / 224],
                        [6, 90 / 224, 90 / 224],
                    ]
                )
            ],
            torch.tensor([[0, 0, 224, 224]]) / 224,
            [
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
            [
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
        ),
        pytest.param(
            1,
            (7, 7),
            [
                torch.tensor(
                    [
                        [6, 224 / 224, 224 / 224],
                    ]
                )
            ],
            torch.tensor([[0, 0, 224, 224]]) / 224,
            [torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])],
            [torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])],
        ),
    ],
)
def test_proto_part_overlap(
    proto_per_class,
    activation_position,
    all_sample_parts_centroids,
    all_sample_bounding_box,
    expected_proto_to_part,
    expected_proto_part_mask,
):
    num_classes = 6
    all_targets = torch.tensor([2])
    # pcs or pss are both fine
    pss = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=proto_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=10,
        uncropped=False,
    )
    all_proto_acts = torch.zeros(1, num_classes * proto_per_class, 14, 14)
    all_proto_acts[:, :, activation_position[0], activation_position[1]] = 1
    all_proto_to_part, all_proto_part_mask = pss.proto2part_and_masks(
        [all_proto_acts],
        [all_targets],
        all_sample_parts_centroids,
        [all_sample_bounding_box],
    )

    assert len(all_proto_to_part) == proto_per_class
    assert len(all_proto_part_mask) == proto_per_class

    for idx in range(proto_per_class):
        assert torch.equal(all_proto_to_part[idx], expected_proto_to_part[idx])
        assert torch.equal(all_proto_part_mask[idx], expected_proto_part_mask[idx])


@pytest.mark.parametrize(
    "proto_per_class,batch_size,activation_position,target",
    [
        pytest.param(1, 3, [(2, 2), (3, 7), (13, 13)], torch.tensor([0, 1, 1])),
        pytest.param(4, 2, [(1, 1), (0, 0)], torch.tensor([0, 1])),
    ],
)
def test_filter_proto_acts(proto_per_class, batch_size, activation_position, target):
    # original author's implementation for checking, same logic
    def process_batch_activations(proto_acts, targets):
        # Select the prototypes belonging to `the ground-truth class of each image
        feature_size = proto_acts.shape[-1]
        proto_indices = (
            (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        )
        # The indexes of prototypes belonging to the ground-truth class of each image
        proto_indices += torch.arange(proto_per_class).to(targets.device)
        proto_indices = proto_indices[:, :, None, None].repeat(
            1, 1, feature_size, feature_size
        )
        proto_acts = torch.gather(
            proto_acts, 1, proto_indices
        )  # (B, proto_per_class, fea_size, fea_size)

        return proto_acts.cpu().detach(), targets.cpu()

    num_classes = 3
    part_num = 10
    # pcs or pss are both fine
    pss = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=proto_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=part_num,
    )
    all_proto_acts = torch.zeros(batch_size, num_classes * proto_per_class, 14, 14)
    for i in range(batch_size):
        all_proto_acts[i, :, activation_position[i][0], activation_position[i][1]] = 1

    proto_acts_selected, targets_selected = pss.filter_proto_acts(
        all_proto_acts, target
    )

    assert torch.equal(targets_selected, target)

    expected_acts_selected, _ = process_batch_activations = process_batch_activations(
        all_proto_acts, target
    )

    assert torch.equal(proto_acts_selected, expected_acts_selected)


@pytest.mark.parametrize(
    "activation_position, batch_size, num_classes, proto_per_class",
    [
        pytest.param([(2, 2), (3, 7), (13, 13)], 33, 1212, 2),
        pytest.param([(1, 1)], 1, 2, 1),
        pytest.param([(0, 0)], 14, 14, 14),
    ],
)
def test_pcs_perfect_score(
    activation_position, batch_size, num_classes, proto_per_class
):
    part_num = len(activation_position)

    pcs = PartConsistencyScore(
        num_classes=num_classes,
        proto_per_class=proto_per_class,
        half_size=36,
        part_thresh=0.8,
        part_num=part_num,
        uncropped=True,
    )

    all_proto_acts = torch.zeros(batch_size, num_classes * proto_per_class, 14, 14)

    all_targets = torch.tensor([0] * batch_size)

    # this ensures activates on a labeled centroid position, gurantees 100% score
    all_sample_parts_centroids = []
    for idx, pos in enumerate(activation_position):
        cx, cy = (pos[0] / 14), (pos[1] / 14)
        all_proto_acts[:, pos[0], pos[1]] = 1
        all_sample_parts_centroids.append([idx, cx, cy])

    all_sample_parts_centroids = [torch.tensor(all_sample_parts_centroids)] * batch_size

    all_sample_bounding_box = torch.tensor([[0, 0, 1, 1]] * batch_size)

    x = 2
    score = 0
    for _ in range(x):
        pcs.update(
            all_proto_acts,
            all_targets,
            all_sample_parts_centroids,
            all_sample_bounding_box,
        )
        score = pcs.compute()

    assert score == 1.0


def test_prototype_ablation_score_known_drop():
    weights = torch.tensor(
        [
            [1.0, 0.5, -0.2],
            [-0.3, 0.8, 1.2],
        ]
    )
    proto_acts = torch.tensor([[[[0.9]], [[0.4]], [[0.2]]]])
    prototype_scores = proto_acts.view(1, 3)
    logits = prototype_scores @ weights.T

    metric = PrototypeAblationScore()
    metric.update(
        proto_acts=proto_acts,
        logits=logits,
        class_connection_weights=weights,
    )

    ablated_scores = torch.tensor([[0.0, 0.4, 0.2]])
    ablated_logits = ablated_scores @ weights.T
    original_confidence = torch.softmax(logits, dim=1)[0, 0]
    ablated_confidence = torch.softmax(ablated_logits, dim=1)[0, 0]
    expected_drop = original_confidence - ablated_confidence

    assert torch.isclose(metric.compute(), expected_drop)


def test_prototype_ablation_score_perfect_score():
    weights = torch.tensor(
        [
            [100.0, -100.0],
            [-100.0, 100.0],
        ]
    )
    proto_acts = torch.tensor([[[[1.0]], [[0.95]]]])
    prototype_scores = proto_acts.view(1, 2)
    logits = prototype_scores @ weights.T

    metric = PrototypeAblationScore()
    metric.update(
        proto_acts=proto_acts,
        logits=logits,
        class_connection_weights=weights,
    )

    assert metric.compute() > 0.99


def test_prototype_ablation_unique_count():
    weights = torch.tensor(
        [
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1],
        ]
    )
    proto_acts = torch.tensor(
        [
            [[[0.9]], [[0.2]], [[0.1]]],
            [[[0.1]], [[0.8]], [[0.2]]],
        ]
    )
    prototype_scores = proto_acts.view(2, 3)
    logits = prototype_scores @ weights.T

    metric = PrototypeAblationUniqueCount()
    metric.update(
        proto_acts=proto_acts,
        logits=logits,
        class_connection_weights=weights,
    )

    assert metric.compute() == 2

# ──────────────────────────────────────────────────────────────────────────────
# PrototypeSelectivityScore tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "batch_size, num_classes, proto_per_class",
    [
        pytest.param(50, 10, 5),
        pytest.param(20, 4, 3),
        pytest.param(100, 2, 10),
    ],
)
def test_selectivity_uniform_is_zero(batch_size, num_classes, proto_per_class):
    """
    A prototype that fires equally on every image should have selectivity 0.
    """
    sel = PrototypeSelectivityScore(
        num_classes=num_classes,
        part_num=1,
        proto_per_class=proto_per_class,
    )

    # Same activation map, identical scores → fires equally on every image
    all_proto_acts = torch.ones(
        batch_size, num_classes * proto_per_class, 7, 7
    )
    all_targets = torch.zeros(batch_size, dtype=torch.long)

    sel.update(all_proto_acts, all_targets)
    score = sel.compute()

    assert abs(score.item()) < 1e-4, (
        f"uniform activation should give selectivity 0, got {score.item()}"
    )


@pytest.mark.parametrize(
    "batch_size, num_classes, proto_per_class",
    [
        pytest.param(20, 4, 3),
        pytest.param(50, 10, 5),
    ],
)
def test_selectivity_one_hot_is_one(batch_size, num_classes, proto_per_class):
    """
    A prototype with all activation mass on a single image should have
    selectivity = 1.
    """
    sel = PrototypeSelectivityScore(
        num_classes=num_classes,
        part_num=1,
        proto_per_class=proto_per_class,
    )

    n_protos = num_classes * proto_per_class
    # Every prototype fires only on image 0 (peak = 5.0 there, 0 elsewhere)
    all_proto_acts = torch.zeros(batch_size, n_protos, 7, 7)
    all_proto_acts[0, :, 3, 3] = 5.0  # only image 0 has any activation
    all_targets = torch.zeros(batch_size, dtype=torch.long)

    sel.update(all_proto_acts, all_targets)
    score = sel.compute()

    assert abs(score.item() - 1.0) < 1e-4, (
        f"one-hot activation should give selectivity 1, got {score.item()}"
    )


def test_selectivity_partial_concentration():
    """
    A prototype that fires equally on 10% of images should give selectivity
    1 - log(0.1*N) / log(N), which is a known closed-form value.
    """
    batch_size = 100
    num_classes = 4
    proto_per_class = 2

    sel = PrototypeSelectivityScore(
        num_classes=num_classes,
        part_num=1,
        proto_per_class=proto_per_class,
    )

    n_protos = num_classes * proto_per_class
    all_proto_acts = torch.zeros(batch_size, n_protos, 7, 7)
    # Each prototype fires equally on the first 10 images, zero elsewhere.
    all_proto_acts[:10, :, 3, 3] = 1.0
    all_targets = torch.zeros(batch_size, dtype=torch.long)

    sel.update(all_proto_acts, all_targets)
    score = sel.compute()

    # Expected: 1 - log(10) / log(100) = 0.5
    expected = 1 - math.log(10) / math.log(100)
    assert abs(score.item() - expected) < 1e-4, (
        f"expected {expected:.4f}, got {score.item()}"
    )


def test_selectivity_accumulates_across_batches():
    """
    Calling update() twice with the same data should give the same result
    as concatenating and updating once — the metric must accumulate
    correctly across batches.
    """
    batch_size = 30
    num_classes = 4
    proto_per_class = 2
    n_protos = num_classes * proto_per_class

    # Build a non-trivial activation pattern
    torch.manual_seed(0)
    all_proto_acts = torch.rand(batch_size, n_protos, 7, 7)
    all_targets = torch.zeros(batch_size, dtype=torch.long)

    # Single-update reference
    sel_ref = PrototypeSelectivityScore(
        num_classes=num_classes, part_num=1, proto_per_class=proto_per_class,
    )
    sel_ref.update(all_proto_acts, all_targets)
    ref = sel_ref.compute()

    # Two-update version with same total data — split in half
    sel_split = PrototypeSelectivityScore(
        num_classes=num_classes, part_num=1, proto_per_class=proto_per_class,
    )
    sel_split.update(all_proto_acts[:15], all_targets[:15])
    sel_split.update(all_proto_acts[15:], all_targets[15:])
    split = sel_split.compute()

    assert abs(ref.item() - split.item()) < 1e-5, (
        f"batched updates differ: {ref.item()} vs {split.item()}"
    )