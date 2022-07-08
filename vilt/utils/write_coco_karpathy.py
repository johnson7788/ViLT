import json
import os
import pandas as pd
import pyarrow as pa
import random
from tqdm import tqdm
from glob import glob
from collections import defaultdict

random.seed(0)

def path2rest(path, iid2captions, iid2split):
    """
    # 读取单张图片，返回图片的二级制格式，图片的字幕列表，图片的名字，图片的所属的数据集名称
    path: 单张图片
    iid2split: 12万张图片数据集对应的名称
    iid2captions:12万张图片对应的字幕
    """
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(root, dataset_root, part_test2train=True):
    """
    part_test2train: 如果我们没有下载训练集的数据，使用部分测试集的数据转成训练集的，方便测试
    """
    with open(f"{root}/karpathy/dataset_coco.json", "r") as fp:
        captions = json.load(fp)
    # 获取图片的信息
    captions = captions["images"]
    # iid2captions: 共12万张图片
    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        data_type = cap["split"]
        if part_test2train and data_type == "test":
            #50%的概率转换成train
            if random.random() >0.5:
                data_type = "train"
        iid2split[filename] = data_type #这条数据集是属于哪个数据集的 'test或val或train'
        for c in cap["sentences"]:  #每个图片的5个描述的句子
            iid2captions[filename].append(c["raw"])  # 我们只要原始的text描述
    # 获取所有的图片的路径
    train_images = list(glob(f"{root}/train2014/*.jpg"))
    val_test_images = list(glob(f"{root}/val2014/*.jpg"))
    if part_test2train:
        assert len(train_images) == 0, f"如果用部分测试集转换成训练集，则训练集必须为空"
    paths = train_images + val_test_images
    random.shuffle(paths)
    # caption_paths: 40504个图片, 里面是每个图片的路径
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("图片的数量和图片字幕的数量相等")
    else:
        print("图片的数量和图片字幕的数量不相等")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "restval", "test"]:
        #从所有数据中，过滤出所属的数据集
        batches = [b for b in bs if b[-1] == split]
        print(f"共有{len(batches)}个图片属于{split}数据集")
        # 生成一个dataframe格式
        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )
        # 转成arrow格式
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/coco_caption_karpathy_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

if __name__ == '__main__':
    make_arrow(root="/home/wac/johnson/johnson/data/coco/root", dataset_root="/home/wac/johnson/johnson/data/coco/arrow",part_test2train=True)
    """
    共有2532个图片属于train数据集
    共有5000个图片属于val数据集
    共有30504个图片属于restval数据集
    共有2468个图片属于test数据集
    4.7G Jul  8 11:33 coco_caption_karpathy_restval.arrow
    386M Jul  8 11:33 coco_caption_karpathy_test.arrow
    399M Jul  8 11:33 coco_caption_karpathy_train.arrow
    777M Jul  8 11:33 coco_caption_karpathy_val.arrow
    """