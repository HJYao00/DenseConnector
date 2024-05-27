import torch
import torch.nn as nn

def dense_connector_sti(image_features, image_forward_outs, is_siglip=True):
    avg_pooling_k8 = nn.AvgPool1d(kernel_size=8, stride=8)
    if not is_siglip:
        image_features_1 = image_forward_outs.hidden_states[7][:, 1:].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, 1:].to(image_features.dtype)
    else:
        image_features_1 = image_forward_outs.hidden_states[7][:, :].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, :].to(image_features.dtype)
    image_features_1 = avg_pooling_k8(image_features_1.permute(0, 2, 1)).permute(0, 2, 1)
    image_features_2 = avg_pooling_k8(image_features_2.permute(0, 2, 1)).permute(0, 2, 1)
    return torch.cat([image_features_1, image_features_2], dim=-2)

def dense_connector_sci(image_features,image_forward_outs, is_siglip=True):
    if not is_siglip:
        image_features_1 = image_forward_outs.hidden_states[7][:, 1:].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, 1:].to(image_features.dtype)
    else:
        image_features_1 = image_forward_outs.hidden_states[7][:, :].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, :].to(image_features.dtype)
    return torch.cat([image_features_1, image_features_2], dim=-1)

def dense_connector_dci(image_features,image_forward_outs, is_siglip=True):
    image_features_1 = []
    image_features_2 = []
    if not is_siglip:
        for i in range(0, 12):
            image_features_1.append(image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype))
        image_features_1 = torch.stack(image_features_1, dim=0)
        image_features_1 = torch.sum(image_features_1, dim=0) / 12
        for i in range(12, 24):
            image_features_2.append(image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype))
        image_features_2 = torch.stack(image_features_2, dim=0)
        image_features_2 = torch.sum(image_features_2, dim=0) / 12
    else:
        for i in range(0, 13):
            image_features_1.append(image_forward_outs.hidden_states[i][:, :].to(image_features.dtype))
        image_features_1 = torch.stack(image_features_1, dim=0)
        image_features_1 = torch.sum(image_features_1, dim=0) / 13
        for i in range(13, 26):
            image_features_2.append(image_forward_outs.hidden_states[i][:, :].to(image_features.dtype))
        image_features_2 = torch.stack(image_features_2, dim=0)
        image_features_2 = torch.sum(image_features_2, dim=0) / 13
    return torch.cat([image_features_1, image_features_2], dim=-1)


def dense_connector(image_features, image_forward_outs, is_siglip=True, mm_dense_connector_type='dci'):
    
    if mm_dense_connector_type == 'sti':
        image_features_dc = dense_connector_sti(image_features, image_forward_outs, is_siglip)
        image_features = torch.cat((image_features, image_features_dc), dim=-2)
    elif mm_dense_connector_type == 'sci':
        image_features_dc = dense_connector_sci(image_features, image_forward_outs, is_siglip)
        image_features = torch.cat((image_features, image_features_dc), dim=-1)
    elif mm_dense_connector_type == 'dci':
        image_features_dc = dense_connector_dci(image_features, image_forward_outs, is_siglip)
        image_features = torch.cat((image_features, image_features_dc), dim=-1)
    else:
        raise NotImplementedError()
    
    return image_features
