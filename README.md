<div align="center">
<br/>
<br/>
  <h1 align="center">
    PointCloudIntelligVis - 新疆大学 - 点云智能感知和三维可视化系统
  </h1>



</div>

#### 项目简介
结合Flask框架和WebGL技术实现了一个点云智能感知和三维可视化系统。该系统能够实现点云数据的批量导入、点云实时分类及分割、点云数据的3D预览及编辑等功能。该系统界面友好，操作简单高效，保证了良好的用户体验。

#### 项目安装和运行

```bash
# 安 装
pip install -r requirement\requirement-dev.txt

# 配 置
.env

```

#### 修改配置

```python
.env
# MySql配置信息
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=PointCloudIntelligVis
MYSQL_USERNAME=root
MYSQL_PASSWORD=root

# Redis 配置
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

```

#### Venv 安装

```bash
python -m venv venv
```

#### 运行项目

```bash
# 初 始 化 数 据 库

flask init
```

执行 flask run 命令启动项目

默认的admin后台密码是：123456
