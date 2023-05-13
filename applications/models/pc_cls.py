import datetime
from applications.extensions import db


class PC_CLS(db.Model):
    __tablename__ = 'pc_cls'
    id = db.Column(db.Integer, primary_key=True, comment="id")
    name = db.Column(db.String(255), nullable=False, comment="文件名")
    mime = db.Column(db.CHAR(50), nullable=False, comment="格式")
    source_href = db.Column(db.String(255), comment="原文件存储路径")
    vis_href = db.Column(db.String(255), comment="可视化HTML存储路径")
    cls_name = db.Column(db.String(255), comment="模型预测的类别名称")
    size = db.Column(db.CHAR(30), nullable=False, comment="文件大小")
    create_time = db.Column(db.DateTime, default=datetime.datetime.now, comment="上传（创建）时间")
