# Dataset Preparation
我们利用7个数据集: Google Conceptual Captions (GCC), Stony Brook University Captions (SBU), Visual Genome (VG), COCO Captions (COCO), Flickr 30K Captions (F30K), Visual Question Answering v2 (VQAv2), and Natural Language for Visual Reasoning 2 (NLVR2).

由于许可证问题，我们不分发数据集。
请自行下载这些数据集。
我们使用`pyarrow`来序列化数据集，转换脚本位于`vilt/utils/write_*.py`中。
请按以下方式组织数据集，并运行`make_arrow`函数，将数据集转换为pyarrow二进制文件。

## GCC
https://ai.google.com/research/ConceptualCaptions/download

GCC提供了图片网址和字幕的元祖，注意有相当一部分网址现在无法访问。
编写你自己的下载脚本并按以下结构组织数据集。

    root
    ├── images_train            
    │   ├── 0000                # First four letters of image name
    │   │   ├── 0000000         # Image Binary
    │   │   ├── 0000001      
    │   │   └── ...
    │   ├── 0001              
    │   │   ├── 0001000      
    │   │   ├── 0001001      
    │   │   └── ...          
    │   └── ...          
    ├── images_val          
    │   ├── 0000              
    │   │   └── ...
    │   └── ...          
    ├── train_annot.json        # List of (image_file_path, caption) tuple
    └── val_annot.json          # List of (image_file_path, caption) tuple

```python
from vilt.utils.write_conceptual_caption import make_arrow
make_arrow(root, arrows_root)
```

## SBU
http://www.cs.virginia.edu/~vicente/sbucaptions/

与GCC类似，SBU也提供了图片网址和字幕的元祖，而且现在有相当一部分网址是无法访问的。
编写你自己的下载脚本，并按以下结构组织数据集。

    root
    ├── images_train            
    │   ├── 0000                # First four letters of image name
    │   │   ├── 0000000         # Image Binary
    │   │   ├── 0000001      
    │   │   └── ...
    │   ├── 0001              
    │   │   ├── 0001000      
    │   │   ├── 0001001      
    │   │   └── ...          
    │   └── ...          
    └── annot.json              # List of (image_file_path, caption) tuple

```python
from vilt.utils.write_sbu import make_arrow
make_arrow(root, arrows_root)
```

## VG
http://visualgenome.org/api/v0/api_home.html

Download [image part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [image part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) and [region descriptions](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip)

    root
    ├── images            
    │   ├── VG_100K                  
    │   │   ├── 10.jpg        
    │   │   ├── 107899.jpg      
    │   │   └── ...
    │   ├── VG_100K_2              
    │   │   ├── 1.jpg      
    │   │   ├── 100.jpg      
    │   │   └── ...          
    │   └── ...          
    └── annotations         
        └── region_descriptions.json

```python
from vilt.utils.write_vg import make_arrow
make_arrow(root, arrows_root)
```

## COCO
https://cocodataset.org/#download

Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...          
    └── karpathy
        └── dataset_coco.json

```python
from vilt.utils.write_coco_karpathy import make_arrow
make_arrow(root, arrows_root)
```

## F30K
http://bryanplummer.com/Flickr30kEntities/

Sign [flickr images request form](https://forms.illinois.edu/sec/229675) and download [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

    root
    ├── flickr30k-images            
    │   ├── 1000092795.jpg
    |   └── ...
    └── karpathy
        └── dataset_flickr30k.json

```python
from vilt.utils.write_f30k_karpathy import make_arrow
make_arrow(root, arrows_root)
```

## VQAv2
https://visualqa.org/download.html

Download COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip))

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...  
    ├── test2015              
    |   ├── COCO_test2015_000000000001.jpg
    |   └── ...         
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    ├── v2_OpenEnded_mscoco_test2015_questions.json
    ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json

```python
from vilt.utils.write_vqa import make_arrow
make_arrow(root, arrows_root)
```

## NLVR2
Clone the [repository](https://github.com/lil-lab/nlvr) and sign the [request form](https://goo.gl/forms/yS29stWnFWzrDBFH3) to download the images.

    root
    ├── images/train           
    │   ├── 0                  
    │   │   ├── train-10108-0-img0.png   
    │   │   └── ...
    │   ├── 1                  
    │   │   ├── train-10056-0-img0.png       
    │   │   └── ...
    │   └── ...
    ├── dev       
    │   ├── dev-0-0-img0.png
    |   └── ...
    ├── test1     
    │   ├── test1-0-0-img0.png
    |   └── ...
    ├── nlvr
    ├── nlvr2
    └── README.md

```python
from vilt.utils.write_nlvr2 import make_arrow
make_arrow(root, arrows_root)
```
