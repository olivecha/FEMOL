import FEMOL
import gif
gif.options.matplotlib["dpi"] = 150


def optimization_to_gif(mesh):
    """
    Creates a gif from the mesh optimization result
    """
    keys = mesh._get_Xi_keys()

    @gif.frame
    def plot_frame(key):
        mesh.plot.cell_data(key)

    frames = []
    for key in keys:
        frame = plot_frame(key)
        frames.append(frame)

    path = ''
    gif_name = path + 'topopt_' + FEMOL.utils.unique_time_string() + '.gif'
    fps = 20  # frames/s

    gif.save(frames, gif_name, duration=len(frames) / fps, unit="s", between="startend")