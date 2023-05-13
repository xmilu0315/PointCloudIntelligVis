from applications.extensions import ma
from marshmallow import fields


class PcSegOutSchema(ma.Schema):
    id = fields.Integer()
    name = fields.Str()
    mime = fields.Str()
    source_href = fields.Str()
    xyzlArray_href = fields.Str()
    vis_href = fields.Str()
    num_part = fields.Str()
    size = fields.Str()
    create_time = fields.DateTime()

