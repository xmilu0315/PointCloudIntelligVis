import datetime
from applications.extensions import db


class PC_CLS_PTH(db.Model):
    __tablename__ = 'pc_cls_pth'
    id = db.Column(db.Integer, primary_key=True, comment="id")
    name = db.Column(db.String(255), nullable=False, comment="")
    size = db.Column(db.String(255), nullable=True, comment="")
    net_no = db.Column(db.String(255), nullable=False, comment="")
    dataset = db.Column(db.String(255), nullable=True, comment="")
    oa = db.Column(db.String(255), nullable=True, comment="")
    macc = db.Column(db.String(255), nullable=True, comment="")
    epoch = db.Column(db.String(255), nullable=True, comment="")
    lr = db.Column(db.String(255), nullable=True, comment="")
    optimizer = db.Column(db.String(255), nullable=True, comment="")
    loss_func = db.Column(db.String(255), nullable=True, comment="")
    create_time = db.Column(db.DateTime, default=datetime.datetime.now, comment="上传（创建）时间")
    pth_path = db.Column(db.String(255), nullable=False, comment="")
    in_used = db.Column(db.Integer, nullable=False, comment="")
