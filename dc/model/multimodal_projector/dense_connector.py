import torch

def dense_connector_sti(image_features, image_forward_outs, is_siglip=True):
    if not is_siglip:
        image_features_1 = image_forward_outs.hidden_states[7][:, 1:].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, 1:].to(image_features.dtype)
    else:
        image_features_1 = image_forward_outs.hidden_states[7][:, :].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, :].to(image_features.dtype)
    image_features_1 = self.get_model().avg_pooling_k8(image_features_1.permute(0, 2, 1)).permute(0, 2, 1)
    image_features_2 = self.get_model().avg_pooling_k8(image_features_2.permute(0, 2, 1)).permute(0, 2, 1)
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
    if not is_siglip:
        image_features_1 = image_forward_outs.hidden_states[7][:, 1:].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, 1:].to(image_features.dtype)
        for i in range(0, 12):
            if i == 0:
                image_features_1 = image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype)
            else:
                image_features_1 += image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype)
        for i in range(12, 24):
            if i == 12:
                image_features_2 = image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype)
            else:
                image_features_2 += image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype)
        image_features_1 = image_features_1 / 12
        image_features_2 = image_features_2 / 12
    else:
        image_features_1 = image_forward_outs.hidden_states[7][:, :].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, :].to(image_features.dtype)
        for i in range(0, 13):
            if i == 0:
                image_features_1 = image_forward_outs.hidden_states[i][:, :].to(image_features.dtype)
            else:
                image_features_1 += image_forward_outs.hidden_states[i][:, :].to(image_features.dtype)
        for i in range(13, 26):
            if i == 13:
                image_features_2 = image_forward_outs.hidden_states[i][:, :].to(image_features.dtype)
            else:
                image_features_2 += image_forward_outs.hidden_states[i][:, :].to(image_features.dtype)
        image_features_1 = image_features_1 / 13
        image_features_2 = image_features_2 / 13
    return torch.cat([image_features_1, image_features_2], dim=-1)