import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SteeringVectorFeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor for steering vectors that respects
    spatial (microphone) and frequency structure
    Enhanced with AoA prediction capability for pretraining
    """
    def __init__(self, input_channels=2, num_angles=181, pretrain_mode=False):
        super().__init__()
        
        self.pretrain_mode = pretrain_mode
        self.num_angles = num_angles  # e.g., 181 for -90° to +90° in 1° steps
        
        # Frequency-domain convolutions (along freq axis)
        self.freq_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(7, 1), padding=(3, 0))
        self.freq_conv2 = nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0))
        self.freq_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        
        # Spatial-domain convolutions (along microphone axis)
        self.spatial_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(1, 5), padding=(0, 2))
        self.spatial_conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        
        # Combined spatio-frequency convolutions
        self.combined_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.combined_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # Additional layers for better feature learning
        self.freq_conv4 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0))
        self.spatial_conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.combined_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        
        # Activation and normalization
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        
        # Batch normalization for better training stability
        self.bn_freq1 = nn.BatchNorm2d(32)
        self.bn_freq2 = nn.BatchNorm2d(64)
        self.bn_freq3 = nn.BatchNorm2d(128)
        self.bn_freq4 = nn.BatchNorm2d(256)
        
        self.bn_spatial1 = nn.BatchNorm2d(32)
        self.bn_spatial2 = nn.BatchNorm2d(64)
        self.bn_spatial3 = nn.BatchNorm2d(128)
        
        self.bn_combined1 = nn.BatchNorm2d(32)
        self.bn_combined2 = nn.BatchNorm2d(64)
        self.bn_combined3 = nn.BatchNorm2d(128)
        
        # AoA prediction head (only used during pretraining)
        if pretrain_mode:
            self.aoa_head = self._build_aoa_head()
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_aoa_head(self):
        """Build the angle-of-arrival prediction head"""
        # Global pooling to aggregate features
        pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.AdaptiveMaxPool2d((1, 1)),  # Global max pooling
        ])
        
        # Calculate total feature dimension
        # freq: 256, spatial: 128, combined: 128, each with avg+max pooling
        total_features = (256 + 128 + 128) * 2  # 1024 features
        
        # MLP head for AoA prediction
        aoa_head = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Regression head
        )
        
        return nn.ModuleDict({
            'pooling': pooling_layers,
            'classifier': aoa_head
        })
    
    def _initialize_weights(self):
        """Custom weight initialization for acoustic data"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier normal initialization for conv layers
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.Linear):
                # Xavier normal for linear layers
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard batch norm initialization
                nn.init.constant_(module.weight.data, 1.0)
                nn.init.constant_(module.bias.data, 0.0)
    
    def forward(self, x, return_aoa=False):
        """
        Extract multi-scale features from steering vectors
        Args:
            x: [B, 2, 1024, 16] - batch of steering vectors
            return_aoa: bool - whether to return AoA prediction (only in pretrain mode)
        Returns:
            List of feature maps at different scales
            Optional: AoA predictions if return_aoa=True and pretrain_mode=True
        """
        features = []
        
        # Frequency-domain features
        freq_f1 = self.act(self.bn_freq1(self.freq_conv1(x)))
        freq_f2 = self.act(self.bn_freq2(self.freq_conv2(freq_f1)))
        freq_f3 = self.act(self.bn_freq3(self.freq_conv3(freq_f2)))
        freq_f4 = self.act(self.bn_freq4(self.freq_conv4(freq_f3)))
        
        # Spatial-domain features
        spatial_f1 = self.act(self.bn_spatial1(self.spatial_conv1(x)))
        spatial_f2 = self.act(self.bn_spatial2(self.spatial_conv2(spatial_f1)))
        spatial_f3 = self.act(self.bn_spatial3(self.spatial_conv3(spatial_f2)))
        
        # Combined features
        combined_f1 = self.act(self.bn_combined1(self.combined_conv1(x)))
        combined_f2 = self.act(self.bn_combined2(self.combined_conv2(combined_f1)))
        combined_f3 = self.act(self.bn_combined3(self.combined_conv3(combined_f2)))
        
        # Collect features at different scales (for feature matching)
        features = [freq_f1, freq_f2, freq_f3, freq_f4, 
                   spatial_f1, spatial_f2, spatial_f3,
                   combined_f1, combined_f2, combined_f3]
        
        # AoA prediction (only during pretraining)
        aoa_pred = None
        if self.pretrain_mode and return_aoa:
            aoa_pred = self._predict_aoa(freq_f4, spatial_f3, combined_f3)
        
        if return_aoa:
            return features, aoa_pred
        else:
            return features
    
    def _predict_aoa(self, freq_feat, spatial_feat, combined_feat):
        """Predict angle of arrival from high-level features"""
        # Global pooling for each feature type
        pooled_features = []
        
        for feat in [freq_feat, spatial_feat, combined_feat]:
            # Average pooling
            avg_pool = self.aoa_head['pooling'][0](feat)
            avg_pool = avg_pool.view(avg_pool.size(0), -1)
            
            # Max pooling
            max_pool = self.aoa_head['pooling'][1](feat)
            max_pool = max_pool.view(max_pool.size(0), -1)
            
            # Concatenate avg and max pooling
            pooled_features.append(torch.cat([avg_pool, max_pool], dim=1))
        
        # Concatenate all features
        combined_features = torch.cat(pooled_features, dim=1)
        
        # Apply dropout
        combined_features = self.dropout(combined_features)
        
        # Get AoA prediction
        aoa_logits = self.aoa_head['classifier'](combined_features)
        
        return aoa_logits
    
    def freeze_backbone(self):
        """Freeze backbone weights for fine-tuning"""
        for name, param in self.named_parameters():
            if 'aoa_head' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights"""
        for param in self.parameters():
            param.requires_grad = True


class SteeringVectorFeatureMatchingLoss(nn.Module):
    """
    Feature matching loss specifically designed for steering vectors
    Now supports pretrained feature extractor
    """
    def __init__(self, input_channels=2, weights=None, pretrained_extractor=None):
        super().__init__()
        
        if pretrained_extractor is not None:
            # Use pretrained feature extractor
            self.feature_extractor = pretrained_extractor
            # Set to non-pretrain mode for feature matching
            self.feature_extractor.pretrain_mode = False
            # Freeze the pretrained weights
            self.feature_extractor.freeze_backbone()
        else:
            # Use fresh feature extractor
            self.feature_extractor = SteeringVectorFeatureExtractor(input_channels, pretrain_mode=False)
        
        # Default weights for different feature scales (updated for more features)
        if weights is None:
            # freq: 4 features, spatial: 3 features, combined: 3 features
            self.weights = [1.0, 1.0, 1.0, 1.2,  # frequency features
                           0.8, 0.8, 1.0,        # spatial features  
                           1.2, 1.2, 1.2]        # combined features
        else:
            self.weights = weights
            
        self.criterion = nn.L1Loss()
        
    def forward(self, generated, target):
        """
        Compute feature matching loss
        Args:
            generated: [B, 2, 1024, 16] - generated steering vectors
            target: [B, 2, 1024, 16] - target steering vectors
        """
        with torch.no_grad():
            target_features = self.feature_extractor(target)
            
        generated_features = self.feature_extractor(generated)
        
        total_loss = 0
        for i, (gen_feat, target_feat) in enumerate(zip(generated_features, target_features)):
            feat_loss = self.criterion(gen_feat, target_feat)
            total_loss += self.weights[i] * feat_loss
            
        return total_loss / len(generated_features)
    
    def load_pretrained_extractor(self, checkpoint_path):
        """Load pretrained feature extractor weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        self.feature_extractor.pretrain_mode = False
        self.feature_extractor.freeze_backbone()
        print(f"Loaded pretrained feature extractor from {checkpoint_path}")


# Utility function to convert angles to class indices
def angle_to_class(angle_degrees, min_angle=0, max_angle=360, num_classes=360):
    """Convert angle in degrees to class index"""
    angle_normalized = (angle_degrees - min_angle) / (max_angle - min_angle)
    class_idx = torch.clamp(angle_normalized * (num_classes - 1), 0, num_classes - 1)
    return class_idx.long()

def class_to_angle(class_idx, min_angle=360, max_angle=360, num_classes=360):
    """Convert class index to angle in degrees"""
    angle_normalized = class_idx.float() / (num_classes - 1)
    angle_degrees = angle_normalized * (max_angle - min_angle) + min_angle
    return angle_degrees