from applications.extensions import ma
from marshmallow import fields


class PcSegPthOutSchema(ma.Schema):
    id = fields.Integer()
    name = fields.Str()
    size = fields.Str()
    net_no = fields.Str()
    dataset = fields.Str()
    ins_miou = fields.Str()
    cls_miou = fields.Str()
    epoch = fields.Str()
    lr = fields.Str()
    optimizer = fields.Str()
    loss_func = fields.Str()
    create_time = fields.Str()
    pth_path = fields.Str()
    in_used = fields.Integer()
