from applications.extensions import ma
from marshmallow import fields


class PcClsOutSchema(ma.Schema):
    id = fields.Integer()
    name = fields.Str()
    mime = fields.Str()
    source_href = fields.Str()
    vis_href = fields.Str()
    cls_name = fields.Str()
    size = fields.Str()
    create_time = fields.DateTime()

