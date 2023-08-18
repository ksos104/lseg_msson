import torch
import clip


from modules_lseg.models.lseg_vit_zs import _make_pretrained_clip_vitl16_384

def main():
    label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    texts = []
    for class_i in range(len(label_list)):
        text = clip.tokenize(label_list[class_i])
        texts.append(text)
        
    clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(pretrained=True)
    
    text_features = [clip_pretrained.encode_text(text.cuda()) for text in texts]
    text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
    
    text_features = torch.cat(text_features)
    
    affinity_map = text_features @ text_features.t()
    
    top3_list = affinity_map.sort(descending=True)[1][:,1:4]
    
    for idx, cls_name in enumerate(label_list):
        print(cls_name, ": ", label_list[top3_list[idx, 0]], ", ", label_list[top3_list[idx, 1]], ", ", label_list[top3_list[idx, 2]])
    
if __name__=="__main__":
    main()