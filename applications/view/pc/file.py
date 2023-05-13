import os
import shutil

import numpy as np
from flask import Blueprint, request, render_template, jsonify, current_app
from datetime import datetime

from applications.common.utils.http import fail_api, success_api, table_api
from applications.common.utils.rights import authorize
from applications.extensions import db
from applications.models import PC_CLS
from .network2.cls.net import ply2classname
from .network2.seg.net import ply2xyzlArray
from werkzeug.utils import secure_filename
from .ply2html import ply_cls2html, ply_seg2html

from sqlalchemy import desc
from applications.extensions import db
from applications.models import PC_CLS
from applications.models import PC_SEG
from applications.models import PC_CLS_PTH
from applications.models import PC_SEG_PTH
from applications.schemas import PcClsOutSchema
from applications.schemas import PcSegOutSchema
from applications.schemas import PcClsPthOutSchema
from applications.schemas import PcSegPthOutSchema
from applications.common.curd import model_to_dicts

pc_file = Blueprint('pcFile', __name__, url_prefix='/pc/file')


###################################################################################################
#                                分类任务
###################################################################################################

# #自定义的点云文件上传API

@pc_file.get('/upload_cls_pc')
def upload_cls_pc():
    return render_template('pc/cls_pc_add.html')


#   上传接口
@pc_file.post('/upload_cls_pc_proc')
# @authorize("admin:file:add", log=True)
def upload_cls_pc_proc():
    if 'file' in request.files:
        pc = request.files['file']

        base_name, ext = os.path.splitext(secure_filename(pc.filename))
        new_name = base_name + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))

        source_path = f'/{current_app.config.get("UPLOADED_CLS_PC_SOURCE_DEST")}/{new_name}{ext}'.replace('\\', '/')
        vis_url = f'/{current_app.config.get("UPLOADED_CLS_PC_VIS_DEST")}/{new_name}.html'.replace('\\', '/')

        pc.save('.' + source_path)  # 保存上传的点云原始文件
        ply_cls2html('.' + source_path, '.' + vis_url)  # 创建可视化HTML
        class_name = ply2classname('.' + source_path)  # 预测类别标签

        # 写数据库
        pc_cls = PC_CLS(name=new_name, mime=ext, source_href=source_path, vis_href=vis_url, cls_name=class_name,
                        size=os.path.getsize('.' + source_path))
        db.session.add(pc_cls)
        db.session.commit()

        res = {
            "message": f'检测结果为：<span style="font-weight:bold;color:green">{class_name}</span> ，正在为您创建三维可视化预览，请稍后~',
            "code": 0,
            "success": True,
            "data": {"html_src": vis_url,
                     "class_name": class_name,
                     "file_name": pc.filename,
                     },
        }
        return jsonify(res)
    return fail_api()


#  点云管理
@pc_file.get('/index_cls')
# @authorize("admin:file:main", log=True)
def index_cls():
    return render_template('pc/cls_pc.html')


#  点云数据
@pc_file.get('/table_pc_cls')
# @authorize("admin:file:main", log=True)
def table_pc_cls():
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)

    # data, count = upload_curd.get_photo(page=page, limit=limit)
    pc_cls = PC_CLS.query.order_by(desc(PC_CLS.create_time)).paginate(page=page, per_page=limit, error_out=False)
    count = PC_CLS.query.count()
    data = model_to_dicts(schema=PcClsOutSchema, data=pc_cls.items)

    return table_api(data=data, count=count)


#    点云删除
@pc_file.route('/delete_pc_cls', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def delete_pc_cls():
    _id = request.form.get('id')

    source_href = PC_CLS.query.filter_by(id=_id).first().source_href
    vis_href = PC_CLS.query.filter_by(id=_id).first().vis_href

    os.remove('.' + source_href)
    os.remove('.' + vis_href)

    res = PC_CLS.query.filter_by(id=_id).delete()
    db.session.commit()

    if res:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


# 点云批量删除
@pc_file.route('/batchRemove_pc_cls', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def batch_remove_pc_cls():
    ids = request.form.getlist('ids[]')
    pc_cls_name = PC_CLS.query.filter(PC_CLS.id.in_(ids)).all()

    for each in pc_cls_name:
        os.remove('.' + each.source_href)
        os.remove('.' + each.vis_href)

    pc_cls = PC_CLS.query.filter(PC_CLS.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    if pc_cls:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


###################################################################################################
#                                分割任务
###################################################################################################
# 自定义的点云文件上传API

@pc_file.get('/upload_seg_pc')
def upload_seg_pc():
    return render_template('pc/seg_pc_add.html')


#   上传接口
@pc_file.post('/upload_seg_pc_proc')
# @authorize("admin:file:add", log=True)
def upload_seg_pc_proc():
    if 'file' in request.files:
        pc = request.files['file']

        base_name, ext = os.path.splitext(secure_filename(pc.filename))
        new_name = base_name + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))

        source_path = f'/{current_app.config.get("UPLOADED_SEG_PC_SOURCE_DEST")}/{new_name}{ext}'.replace('\\', '/')
        xyzlArray_path = f'/{current_app.config.get("UPLOADED_SEG_PC_XYZLARRAY_DEST")}/{new_name}.npy'.replace('\\',
                                                                                                               '/')

        vis_url = f'/{current_app.config.get("UPLOADED_SEG_PC_VIS_DEST")}/{new_name}.html'.replace('\\', '/')

        pc.save('.' + source_path)  # 保存上传的点云原始文件

        xyzlArray = ply2xyzlArray('.' + source_path)
        np.save('.' + xyzlArray_path, xyzlArray)

        ply_seg2html(xyzlArray, '.' + vis_url)  # 创建可视化HTML
        num_part = len(np.unique(xyzlArray[:, -1]).tolist())

        # 写数据库
        pc_seg = PC_SEG(name=new_name, mime=ext, source_href=source_path, xyzlArray_href=xyzlArray_path,
                        vis_href=vis_url, num_part=num_part,
                        size=os.path.getsize('.' + source_path))
        db.session.add(pc_seg)
        db.session.commit()

        res = {
            "message": f'分割完成，共分割为<span style="font-weight:bold;color:green">{num_part}</span>个部分，正在为您创建三维可视化预览，请稍后~',
            "code": 0,
            "success": True,
            "data": {"file_name": pc.filename,
                     "html_src": vis_url,
                     "num_part": num_part},
        }
        return jsonify(res)
    return fail_api()


#  点云管理
@pc_file.get('/index_seg')
# @authorize("admin:file:main", log=True)
def index_seg():
    return render_template('pc/seg_pc.html')


#  点云数据
@pc_file.get('/table_pc_seg')
# @authorize("admin:file:main", log=True)
def table_pc_seg():
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)

    pc_seg = PC_SEG.query.order_by(desc(PC_SEG.create_time)).paginate(page=page, per_page=limit, error_out=False)
    count = PC_SEG.query.count()
    data = model_to_dicts(schema=PcSegOutSchema, data=pc_seg.items)

    return table_api(data=data, count=count)


#    点云删除
@pc_file.route('/delete_pc_seg', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def delete_pc_seg():
    _id = request.form.get('id')

    source_href = PC_SEG.query.filter_by(id=_id).first().source_href
    xyzlArray_href = PC_SEG.query.filter_by(id=_id).first().xyzlArray_href
    vis_href = PC_SEG.query.filter_by(id=_id).first().vis_href
    os.remove('.' + source_href)
    os.remove('.' + xyzlArray_href)
    os.remove('.' + vis_href)

    res = PC_SEG.query.filter_by(id=_id).delete()
    db.session.commit()

    if res:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


# 点云批量删除
@pc_file.route('/batchRemove_pc_seg', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def batch_remove_pc_seg():
    ids = request.form.getlist('ids[]')
    pc_seg_name = PC_SEG.query.filter(PC_SEG.id.in_(ids)).all()

    for pc_seg in pc_seg_name:
        os.remove('.' + pc_seg.source_href)
        os.remove('.' + pc_seg.xyzlArray_href)
        os.remove('.' + pc_seg.vis_href)

    pc_seg = PC_SEG.query.filter(PC_SEG.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    if pc_seg:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


###################################################################################################
#                                权重管理--分类
###################################################################################################
# 自定义的权重文件上传API

@pc_file.get('/upload_cls_pc_pth')
def upload_cls_pc_pth():
    return render_template('pc/cls_pc_pth_add.html')


#   权重上传接口
@pc_file.post('/upload_cls_pc_pth_proc')
# @authorize("admin:file:add", log=True)
def upload_cls_pc_pth_proc():
    if 'file' in request.files:
        pc = request.files['file']

        base_name, ext = os.path.splitext(secure_filename(pc.filename))
        new_name = base_name + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))

        pth_path = f'/{current_app.config.get("UPLOADED_NET2_CLS_PTH_DEST")}/{new_name}{ext}'.replace('\\', '/')

        pc.save('.' + pth_path)  # 保存上传的点云原始文件

        # 写数据库
        cls_pth = PC_CLS_PTH(name=new_name, net_no=2, pth_path=pth_path, in_used=0)
        db.session.add(cls_pth)
        db.session.commit()

        res = {
            "message": '权重文件更新成功~！',
            "code": 0,
            "success": True,
            "data": {},
        }
        return jsonify(res)
    return fail_api()


#  权重管理
@pc_file.get('/index_cls_pth')
# @authorize("admin:file:main", log=True)
def index_cls_pth():
    return render_template('pc/cls_pc_pth.html')


#  权重数据
@pc_file.get('/table_pc_cls_pth')
# @authorize("admin:file:main", log=True)
def table_pc_cls_pth():
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)

    pc_cls_pth = PC_CLS_PTH.query.order_by(desc(PC_CLS_PTH.create_time)).paginate(page=page, per_page=limit,
                                                                                  error_out=False)
    count = PC_CLS_PTH.query.count()
    data = model_to_dicts(schema=PcClsPthOutSchema, data=pc_cls_pth.items)

    return table_api(data=data, count=count)


#    权重删除
@pc_file.route('/delete_pc_cls_pth', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def delete_pc_cls_pth():
    _id = request.form.get('id')

    pth_path = PC_CLS_PTH.query.filter_by(id=_id).first().pth_path
    os.remove('.' + pth_path)

    res = PC_CLS_PTH.query.filter_by(id=_id).delete()
    db.session.commit()

    if res:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


# 权重批量删除
@pc_file.route('/batchRemove_pc_cls_pth', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def batch_remove_pc_cls_pth():
    ids = request.form.getlist('ids[]')
    pc_cls_pth_name = PC_CLS_PTH.query.filter(PC_CLS_PTH.id.in_(ids)).all()

    for each in pc_cls_pth_name:
        os.remove('.' + each.pth_path)
        print(f"已成功删除{'.' + each.pth_path}")

    pc_cls_pth = PC_CLS_PTH.query.filter(PC_CLS_PTH.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    if pc_cls_pth:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


# 权重关闭
@pc_file.route('/cls_pth_disable', methods=['GET', 'POST'])
# @authorize("admin:role:edit", log=True)
def cls_pth_disable():
    id = request.json.get('id')
    if id:
        res = PC_CLS_PTH.query.filter_by(id=id).update({"in_used": 0})
        db.session.commit()
        if not res:
            return fail_api(msg="出错啦")
        return success_api(msg="关闭成功")
    return fail_api(msg="数据错误")


# 权重激活
@pc_file.route('/cls_pth_enable', methods=['GET', 'POST'])
# @authorize("admin:role:edit", log=True)
def cls_pth_enable():
    id = request.json.get('id')
    if id:
        pth_path = PC_CLS_PTH.query.filter_by(id=id).first().pth_path
        pth_dir, _ = os.path.split(pth_path)
        shutil.copy(src='.' + pth_path, dst=os.path.join('.' + pth_dir, "in_used.pth"))  # 复制文件

        PC_CLS_PTH.query.update({"in_used": 0})  # 先把所有的都权重禁用，确保只有一个权重启用
        res = PC_CLS_PTH.query.filter_by(id=id).update({"in_used": 1})  # 把选中的权重启用
        db.session.commit()
        if not res:
            return fail_api(msg="出错啦")
        return success_api(msg="启用成功")
    return fail_api(msg="数据错误")


###################################################################################################
#                                权重管理--分割
###################################################################################################
# 自定义的权重文件上传API

@pc_file.get('/upload_seg_pc_pth')
def upload_seg_pc_pth():
    return render_template('pc/seg_pc_pth_add.html')


#   权重上传接口
@pc_file.post('/upload_seg_pc_pth_proc')
# @authorize("admin:file:add", log=True)
def upload_seg_pc_pth_proc():
    if 'file' in request.files:
        pc = request.files['file']

        base_name, ext = os.path.splitext(secure_filename(pc.filename))
        new_name = base_name + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))

        pth_path = f'/{current_app.config.get("UPLOADED_NET2_SEG_PTH_DEST")}/{new_name}{ext}'.replace('\\', '/')

        pc.save('.' + pth_path)  # 保存上传的点云原始文件

        # 写数据库
        seg_pth = PC_SEG_PTH(name=new_name, net_no=2, pth_path=pth_path, in_used=0)
        db.session.add(seg_pth)
        db.session.commit()

        res = {
            "message": '权重文件更新成功~！',
            "code": 0,
            "success": True,
            "data": {},
        }
        return jsonify(res)
    return fail_api()


#  权重管理
@pc_file.get('/index_seg_pth')
# @authorize("admin:file:main", log=True)
def index_seg_pth():
    return render_template('pc/seg_pc_pth.html')


#  权重数据
@pc_file.get('/table_pc_seg_pth')
# @authorize("admin:file:main", log=True)
def table_pc_seg_pth():
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)

    pc_seg_pth = PC_SEG_PTH.query.order_by(desc(PC_SEG_PTH.create_time)).paginate(page=page, per_page=limit,
                                                                                  error_out=False)
    count = PC_SEG_PTH.query.count()
    data = model_to_dicts(schema=PcSegPthOutSchema, data=pc_seg_pth.items)

    return table_api(data=data, count=count)


#    权重删除
@pc_file.route('/delete_pc_seg_pth', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def delete_pc_seg_pth():
    _id = request.form.get('id')

    pth_path = PC_SEG_PTH.query.filter_by(id=_id).first().pth_path
    res = PC_SEG_PTH.query.filter_by(id=_id).delete()
    db.session.commit()
    os.remove('.' + pth_path)

    if res:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


# 权重批量删除
@pc_file.route('/batchRemove_pc_seg_pth', methods=['GET', 'POST'])
# @authorize("admin:file:delete", log=True)
def batch_remove_pc_seg_pth():
    ids = request.form.getlist('ids[]')
    pc_seg_pth_name = PC_SEG_PTH.query.filter(PC_SEG_PTH.id.in_(ids)).all()

    for each in pc_seg_pth_name:
        os.remove('.' + each.pth_path)
        print(f"已成功删除{'.' + each.pth_path}")

    pc_seg_pth = PC_SEG_PTH.query.filter(PC_SEG_PTH.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    if pc_seg_pth:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


# 权重关闭
@pc_file.route('/seg_pth_disable', methods=['GET', 'POST'])
# @authorize("admin:role:edit", log=True)
def seg_pth_disable():
    id = request.json.get('id')
    if id:
        res = PC_SEG_PTH.query.filter_by(id=id).update({"in_used": 0})
        db.session.commit()
        if not res:
            return fail_api(msg="出错啦")
        return success_api(msg="关闭成功")
    return fail_api(msg="数据错误")


# 权重激活
@pc_file.route('/seg_pth_enable', methods=['GET', 'POST'])
# @authorize("admin:role:edit", log=True)
def seg_pth_enable():
    id = request.json.get('id')
    if id:
        pth_path = PC_SEG_PTH.query.filter_by(id=id).first().pth_path
        pth_dir, _ = os.path.split(pth_path)
        shutil.copy(src='.' + pth_path, dst=os.path.join('.' + pth_dir, "in_used.pth"))  # 复制文件

        PC_SEG_PTH.query.update({"in_used": 0})  # 先把所有的都权重禁用，确保只有一个权重启用
        res = PC_SEG_PTH.query.filter_by(id=id).update({"in_used": 1})  # 把选中的权重启用
        db.session.commit()
        if not res:
            return fail_api(msg="出错啦")
        return success_api(msg="启用成功")
    return fail_api(msg="数据错误")
