import hashlib
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from glob import glob
from transforms3d.quaternions import mat2quat, quat2mat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref

from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class CustomDataset(object):
    """custom dataset"""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg
        self.objs = ref.custom.objects
        self.width = ref.custom.width
        self.height = ref.custom.height

        self.dataset_root = ref.custom.dataset_root
        assert osp.exists(self.dataset_root), self.dataset_root

        self.model_dir = ref.custom.model_dir
        self.scene_dir = ref.custom.scene_dir

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)

    # TODO: rewrite this function
    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)
        else:
            logger.info("cached dataset will be saved at {}".format(cache_path))

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  # ######################################################
        # it is slow because of loading and converting masks to rle
        scene_dirs = [x for x in sorted(glob(osp.join(self.scene_dir, '*'))) if osp.isdir(x)]
        for scene_dir in tqdm(scene_dirs):
            scene_id = scene_dir.split('/')[-1]
            int_scene_id = int(scene_id)

            scene_cameras = mmcv.load(osp.join(scene_dir, "scene_cameras.json"))
            annotations = mmcv.load(osp.join(scene_dir, "annotations.json"))

            im_ids = sorted(scene_cameras.keys())
            for im_id in tqdm(im_ids, postfix=f"{int_scene_id}"):
                int_im_id = int(im_id)
                rgb_path = osp.join(scene_dir, f"color/{scene_id}.png")
                depth_path = osp.join(scene_dir, f"depth/{scene_id}.png")
                assert osp.exists(rgb_path)
                assert osp.exists(depth_path)

                scene_im_id = f"{int_scene_id}/{int_im_id}"

                cam = scene_cameras[im_id]
                K = np.array([cam['fx'], 0, cam['ppx'], 0, cam['fy'], cam['ppy'], 0, 0, 1], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam["depth_scale"]  # 10000

                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,  # for evaluation
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "syn_pbr",  # NOTE: has background
                }
                import ipdb; ipdb.set_trace()
                insts = []
                for anno_i, anno in enumerate(annotations[im_id]):
                    # # TODO: support multiple object type
                    # obj_id = anno["obj_id"]
                    # if obj_id not in self.cat_ids:
                    #     continue
                    # cur_label = self.cat2label[obj_id]  # 0-based label
                    cur_label = 0;

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])
                    quat = mat2quat(R).astype("float32")

                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                    bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file = osp.join(
                        scene_dir,
                        "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i),
                    )
                    mask_visib_file = osp.join(
                        scene_dir,
                        "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i),
                    )
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib
                    mask_single = mmcv.imread(mask_visib_file, "unchanged")
                    mask_single = mask_single.astype("bool")
                    area = mask_single.sum()
                    if area < 30:  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)

                    # load mask full
                    mask_full = mmcv.imread(mask_file, "unchanged")
                    mask_full = mask_full.astype("bool")
                    mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)

                    visib_fract = gt_info_dict[str_im_id][anno_i].get("visib_fract", 1.0)

                    xyz_path = osp.join(
                        self.xyz_root,
                        f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl",
                    )
                    # assert osp.exists(xyz_path), xyz_path
                    inst = {
                        "category_id": cur_label,  # 0-based label
                        "bbox": bbox_visib,
                        "bbox_obj": bbox_obj,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": t,
                        "centroid_2d": proj,  # absolute (cx, cy)
                        "segmentation": mask_rle,
                        "mask_full": mask_full_rle,
                        "visib_fract": visib_fract,
                        "xyz_path": xyz_path,
                    }

                    model_info = self.models_info[str(obj_id)]
                    inst["model_info"] = model_info
                    for key in ["bbox3d_and_center"]:
                        inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.model_dir, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.cache_dir, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.model_dir,
                    f"obj_{ref.itodd.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


##########################################################################


SPLITS_CUSTOM = dict(
    custom_train = dict(
        name="custom_train",
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
    ),
)

def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_CUSTOM:
        used_cfg = SPLITS_CUSTOM[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, CustomDataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        ref_key='custom',
        id="custom",  # NOTE: for pvnet to determine module
        objs=ref.custom.objects,
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
    )


def get_available_datasets():
    return list(SPLITS_CUSTOM.keys())


#### tests ###############################################
def test_vis():
    dset_name = sys.argv[1]
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_mmcv(d["file_name"], format="BGR")
        detection_utils.check_image_size(d, img)
        depth = mmcv.imread(d["depth_file"], "unchanged") / d["depth_factor"]

        imH, imW = img.shape[:2]
        grid_show(
            [img[:, :, [2, 1, 0]], depth],
            [f"img:{d['file_name']}", "depth"],
            row=1,
            col=2,
        )


if __name__ == "__main__":
    """Test the  dataset loader.

    python this_file.py dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_mmcv
    from detectron2.data import detection_utils

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
