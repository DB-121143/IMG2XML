# Image to DrawIO (XML) Converter
一键将静态图表（流程图、架构图、技术示意图）转化为 **可编辑DrawIO (mxGraph) XML文件**，基于SAM 3与多模态大模型实现高保真重建，保留原图表细节与逻辑关系，赋能快速二次编辑。

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Image2DrawIO-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/XiangjianYi/Image2DrawIO)
[![CUDA Required](https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)

---

## 🌟 核心优势
### 精准分割与重建
- **SAM 3 驱动**：基于最新分割模型，实现图表元素（形状、箭头、图标、文本块）的像素级精准识别，不漏掉虚线、纹理等细节。
- **矢量形状还原**：自动匹配12+常用图表形状（矩形、菱形、圆柱、云形、平行四边形等），支持填充色与描边色智能区分。

### 智能文本与视觉处理
- **混合OCR引擎**：Azure文档智能定位文本区域 + Qwen/Mistral VLM校正识别结果，支持LaTeX公式转换，杜绝文本幻觉。
- **背景净化**：集成RMBG-2.0模型，自动去除图标、箭头背景，生成透明元素，适配DrawIO编辑场景。
- **箭头保真**：箭头单独提取为透明图层，保留路由逻辑与样式（虚线、粗细），可直接调整位置。

### 高效与扩展性
- **迭代式提取**：VLM主动扫描空白区域，生成补充提示词，避免元素遗漏，提升重建完整性。
- **并发优化**：全局锁保障GPU模型线程安全，LRU缓存复用SAM 3图像嵌入，加速交互式编辑。
- **灵活部署**：支持Web界面可视化操作与命令行批量处理，适配不同使用场景。

---

## 📸 效果演示
### 输入输出高清对比（3组典型场景）
为了更直观展示高保真转换效果，以下提供3组「原始静态图片」与「DrawIO可编辑重建结果」的一一对应对比，所有元素均可单独拖拽、修改样式与文本。

| 场景序号 | 原始静态图表（输入·不可编辑） | DrawIO重建结果（输出·全可编辑） |
|----------|------------------------------|--------------------------------|
| 场景1：基础流程图 | <img src="/static/demo/original_1.jpg" width="400" alt="原始图表1" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_1.png" width="400" alt="重建结果1" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 场景2：多层级架构图 | <img src="/static/demo/original_2.png" width="400" alt="原始图表2" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_2.png" width="400" alt="重建结果2" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 场景3：技术示意图 | <img src="/static/demo/original_3.jpg" width="400" alt="原始图表3" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_3.png" width="400" alt="重建结果3" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 场景3：科学公式图 | <img src="/static/demo/original_4.jpg" width="400" alt="原始图表4" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_4.png" width="400" alt="重建结果4" style="border: 1px solid #eee; border-radius: 4px;"/> |

> ✨ 转换亮点说明：
> 1.  保留原图表的布局逻辑、颜色搭配与元素层级关系
> 2.  形状描边/填充、箭头样式（虚线/粗细）1:1还原
> 3.  文本内容精准识别，支持后续直接编辑与格式调整
> 4.  所有元素独立可选中，支持DrawIO原生模板替换与布局优化

---

## 🚀 快速部署
### 前置依赖
- Python 3.10+
- Node.js & npm（Web前端运行）
- CUDA 11.8+（推荐，SAM 3/RMBG模型加速）

### 安装步骤
1.  克隆仓库并安装Python依赖
    ```bash
    git clone https://github.com/XiangjianYi/Image2DrawIO.git
    cd Image2DrawIO
    ```

2.  模型准备
    - **RMBG-2.0**：从[HuggingFace](https://huggingface.co/briaai/RMBG-2.0)下载`model.onnx`，放入`models/rmbg/`目录。
    - **SAM 3**：下载模型权重后，在`config/config.yaml`中配置权重文件路径。

3.  环境变量配置
    在`flowchart_text/.env`文件中填写API密钥与端点：
    ```env
    # Azure文档智能（文本定位）
    AZURE_ENDPOINT=https://你的资源名.cognitiveservices.azure.com/
    AZURE_API_KEY=你的Azure密钥

    # 多模态LLM（文本识别/公式转换）
    MISTRAL_API_KEY=你的API密钥
    MISTRAL_MODEL=qwen-vl-max
    MISTRAL_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
    ```

### 使用方式
#### 1. Web界面（推荐，可视化操作）
```bash
# 启动后端服务
python server.py

# 启动前端（新终端）
cd frontend
npm install && npm run dev
```

##### 前端网页界面展示
<img src="/static/demo/frontend.png" width="800" alt="Image2DrawIO前端网页截图" style="border: 1px solid #eee; border-radius: 8px; margin: 16px 0;"/>

✨ 前端界面亮点：简洁直观的操作面板、图片上传预览、转换进度实时显示、一键导出DrawIO XML文件

浏览器访问 `https://db121-img2xml.cn/`，上传图片即可完成转换，一键导出DrawIO XML。
> 📌 操作提示：支持拖拽上传/点击选择图片，转换完成后会自动弹出下载弹窗，也可在结果预览区手动点击「导出XML」按钮。

#### 2. 命令行（批量/脚本集成）
```bash
# 单张图片转换
python scripts/run_all.py input/test.jpg --output output/result.xml

# 批量转换（文件夹下所有图片）
python scripts/run_all.py input/ --batch --output output/
```

---

## 📂 项目结构
```
Image2DrawIO/
├── server.py               # 后端API服务（FastAPI）
├── frontend/               # Web前端（React+Vite）
├── scripts/
│   └── run_all.py          # 命令行转换入口（支持批量处理）
├── models/                 # 预训练模型目录
│   └── rmbg/               # RMBG-2.0模型文件
├── config/
│   └── config.yaml         # 模型路径、参数配置
├── flowchart_text/         # OCR与文本处理核心模块
├── docs/                   # 技术文档、API说明
├── input/                  # 测试输入目录
├── output/                 # 转换结果输出目录
└── requirements.txt        # Python依赖清单
```

---

## 📌 开发规划
| 功能模块         | 状态       | 说明                     |
|------------------|------------|--------------------------|
| 核心转换流水线   | ✅ 已完成  | 分割、重建、OCR全流程    |
| 箭头智能连接     | ⚠️ 开发中  | 自动关联箭头与目标形状   |
| DrawIO模板适配   | 📍 规划中  | 支持自定义模板导入       |
| 批量导出优化     | 📍 规划中  | 批量导出为DrawIO文件（.drawio） |
| 本地LLM适配      | 📍 规划中  | 支持本地部署VLM，脱离API |

---

## 🤝 贡献指南
欢迎各类贡献（代码提交、Bug反馈、功能建议）：
1.  Fork本仓库
2.  创建特性分支（`git checkout -b feature/xxx`）
3.  提交修改（`git commit -m 'feat: add xxx'`）
4.  推送分支（`git push origin feature/xxx`）
5.  发起Pull Request

问题反馈：[Issues](https://github.com/XiangjianYi/Image2DrawIO/issues)  
功能建议：[Discussions](https://github.com/XiangjianYi/Image2DrawIO/discussions)

---

## 📄 许可证
本项目基于 [Apache License 2.0](LICENSE) 开源，允许商用与二次开发（保留版权声明）。

---
> 🌟 若本项目对你有帮助，欢迎点亮Star支持！
> 
> [![GitHub stars](https://img.shields.io/github/stars/XiangjianYi/Image2DrawIO?style=social)](https://github.com/XiangjianYi/Image2DrawIO/stargazers)

---
