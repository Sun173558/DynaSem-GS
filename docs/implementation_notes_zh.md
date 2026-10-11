# MonoGS语义建图实现解析

## 分割器实现

### 语义分割器 (`utils/semantic_utils.py`)

语义分割器使用MMSegmentation的Mask2Former模型，在初始化时加载`mask2former_swin-t_8xb2-160k_ade20k-512x512`预训练模型。分割器定义了两类物体：`seg_classes`包含桌子、椅子、沙发等静态物体，`dyna_classes`包含人等动态物体。

```python
def segment_image(self, image):
    result = self.inferencer(image, show=False)
    predictions = result['predictions']
    self.last_predictions = predictions
    return predictions
```

`segment_image()`函数接收RGB图像，返回每个像素的类别ID。分割结果存储在`last_predictions`中供后续函数使用。

### 实例分割器 (`utils/instance_utils.py`)

实例分割器使用YOLO模型进行实例分割，在初始化时可配置模型名称、置信度阈值和输出：

```python
def __init__(self, model_name='yolov8m-seg.pt', verbose=False, conf_threshold=0.5):
    self.model = YOLO(model_name)
    self.verbose = verbose
    self.conf_threshold = conf_threshold
```

分割器定义了目标类别集合`target_classes`，包含人、椅子、沙发、餐桌等物体，以及动态类别`dynamic_classes`（主要是人）。

```python
def segment_image(self, image):
    results = self.model(original_image, verbose=False)
    # 过滤置信度并生成实例掩码
    for i, cls in enumerate(classes):
        if scores[i] > self.conf_threshold:
            # 处理检测结果
```

实例分割器返回带有唯一ID的实例掩码，每个检测到的物体实例都有不同的ID值。

### 掩码生成机制

两种分割器都提供统一的掩码生成接口：

**语义分割器的掩码生成：**
```python
def get_dynamic_mask(self):
    mask = np.zeros(self.last_predictions.shape, dtype=bool)
    for class_name in self.dyna_classes:
        if class_name in self.class_to_idx:
            class_idx = self.class_to_idx[class_name]
            mask[self.last_predictions == class_idx] = True
    return mask
```

**实例分割器的掩码生成：**
```python
def get_dynamic_mask(self):
    mask = np.zeros((h, w), dtype=bool)
    for i, cls in enumerate(self.last_classes):
        class_name = self.get_class_name(cls)
        if class_name in self.dynamic_classes:
            instance_mask = cv2.resize(self.last_masks[i], (w, h)) > 0.5
            mask[instance_mask] = True
    return mask
```

两种分割器都支持获取动态掩码、静态掩码和特定类别掩码，为SLAM系统提供一致的接口。

## 相机对象的语义扩展

### 语义信息集成 (`utils/camera_utils.py`)

相机类在初始化时自动进行分割处理。`init_from_dataset()`函数接收分割器参数，对RGB图像进行分割并存储结果：

```python
if segmentor is not None:
    segmentor.segment_image(color)
    self.dynamic_mask = segmentor.get_dynamic_mask()
    self.semantic_image = segmentor.get_semantic_image()
    self.original_image[:, self.dynamic_mask] = 0  # 动态区域置零
```

系统支持两种分割器：
- **语义分割器**：提供像素级的语义标签，适合密集的语义建图
- **实例分割器**：提供实例级的分割结果，能区分同类别的不同物体实例

关键操作是将动态区域的RGB像素置零，这样后续的特征提取和跟踪就不会受到动态物体影响。

### 数据结构设计

每个相机对象包含完整的多模态信息：RGB图像、深度图、分割图像、动态掩码、相机内外参数。这种设计使得分割信息与几何信息保持同步，便于后续处理。

## 高斯模型的语义特征

### 语义参数存储 (`gaussian_splatting/scene/gaussian_model.py`)

高斯模型增加了语义特征存储：

```python
self._semantic_dc = torch.empty(0)      # 语义直流分量
self._semantic_rest = torch.empty(0)    # 语义球谐系数
```

这些参数与RGB特征`_features_dc`和`_features_rest`采用相同的球谐表示方法，支持视角相关的语义渲染。

### 点云创建过程

`create_pcd_from_image()`函数从RGB-D图像和语义图像创建带语义的点云：

```python
# RGB特征提取
fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1)**2)).float().cuda()
features[:, :3, 0] = fused_color

# 语义特征提取
semantic_color = RGB2SH(torch.from_numpy(pcd.semantics).float().cuda())
semantics = torch.zeros((semantic_color.shape[0], 3, (self.max_sh_degree + 1)**2)).float().cuda()
semantics[:, :3, 0] = semantic_color
```

语义特征从语义图像的RGB值转换而来，通过`RGB2SH`函数转换为球谐系数。每个3D点同时具备RGB和语义两种特征表示。

### 点云扩展机制

`extend_from_pcd_seq()`函数在添加新关键帧时扩展高斯点云：

```python
def extend_from_pcd_seq(self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None):
    fused_point_cloud, features, semantics, scales, rots, opacities = (
        self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap))
    self.extend_from_pcd(fused_point_cloud, features, semantics, scales, rots, opacities, kf_id)
```

新创建的点同时包含几何、RGB和语义信息，通过`extend_from_pcd()`添加到现有点云中。

## 前端语义处理流程

### 主循环集成 (`utils/slam_frontend.py`)

前端主循环`run()`中，每个新帧都通过语义分割器处理：

```python
# 创建带语义的相机对象
viewpoint = Camera.init_from_dataset(dataset, frame_idx, segmentor=self.segmentor)
self.cameras[frame_idx] = viewpoint
```

语义分割在相机创建阶段完成，确保每帧都有对应的语义信息。前端不直接处理语义特征，主要负责数据预处理和传递。

### 关键帧管理

`add_new_keyframe()`函数添加关键帧时，语义信息随相机对象一起传递给后端：

```python
def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
    self.kf_indices.append(cur_frame_idx)
    viewpoint = self.cameras[cur_frame_idx]  # 包含语义信息的相机
    # 传递给后端处理
```

关键帧的选择标准不直接依赖语义信息，但语义预处理（动态区域移除）会影响特征匹配质量。

## 后端语义优化

### 联合损失计算 (`utils/slam_backend.py`)

后端优化的核心是联合计算RGB和语义损失。在`map()`函数中：

```python
# RGB渲染和损失
render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint, opacity)

# 语义渲染和损失
semantic_pkg = render(viewpoint, self.gaussians, self.pipeline_params,
                     self.background, semantic=True)
sem_img = semantic_pkg["render"]
loss_mapping += get_loss_semantic(self.config, sem_img, viewpoint)
```

系统先渲染RGB图像计算几何损失，再渲染语义图像计算语义损失。两个损失相加形成总损失用于优化。

### 语义损失实现 (`utils/slam_utils.py`)

`get_loss_semantic()`函数计算语义损失：

```python
def get_loss_semantic(config, semimg, viewpoint):
    gt_image = viewpoint.semantic_image.cuda()
    rgb_pixel_mask = ~viewpoint.dynamic_mask
    l1_rgb = torch.abs(semimg * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1_rgb.mean()
```

损失计算使用L1范数，比较渲染的语义图像与真实语义图像。动态掩码用于排除动态区域，只在静态区域计算损失。

### 优化过程

优化过程中，RGB和语义特征同时更新：

```python
loss_mapping.backward()
self.gaussians.optimizer.step()
self.gaussians.optimizer.zero_grad(set_to_none=True)
```

梯度反传会同时更新`_features_dc`、`_features_rest`、`_semantic_dc`、`_semantic_rest`等参数，实现RGB和语义特征的联合优化。

## 语义渲染实现

### 渲染器修改 (`submodules/diff-gaussian-rasterization`)

渲染器支持语义模式，通过`semantic=True`参数切换：

```python
# RGB渲染
render_pkg = render(viewpoint, gaussians, pipeline_params, background)

# 语义渲染
semantic_pkg = render(viewpoint, gaussians, pipeline_params, background, semantic=True)
```

语义渲染时，渲染器使用`_semantic_dc`和`_semantic_rest`参数而非RGB特征，输出语义图像而非RGB图像。

### GUI可视化 (`gui/slam_gui.py`)

GUI支持语义可视化，通过复选框切换显示模式：

```python
def rasterise(self, current_cam):
    if self.semantic_shader_chbox.checked:
        rendering_data = render(current_cam, self.gaussian_cur, self.pipe,
                               self.background, semantic=True)
    else:
        rendering_data = render(current_cam, self.gaussian_cur, self.pipe, self.background)
```

用户可以实时切换RGB和语义视图，观察重建的语义一致性。
