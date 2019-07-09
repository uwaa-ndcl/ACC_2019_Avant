import pkgutil
import subprocess
import pose_estimation.directories as dirs

# get path to render script
mod_name = 'pose_estimation.blender.process_renders'
pkg = pkgutil.get_loader(mod_name)
render_script = pkg.get_filename()

# functions
def blender_render(render_dir):
    '''
    call a blender command which will generate renders in render_dir
    '''

    blender_cmd = 'blender --background --python ' + render_script + \
                  ' -- ' + render_dir
    subprocess.run([blender_cmd], shell=True)
