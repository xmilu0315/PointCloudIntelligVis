from flask import Flask

from applications.view.pc.file import pc_file



def register_pc_views(app: Flask):
    app.register_blueprint(pc_file)
