import logging

import numpy as np

from imgui_bundle import imgui, hello_imgui, immvision

import jsonschema

from defined_pipelines import defined_pipelines
from img_stats import compute_img_stats
from imgui_logger import ImGuiHandler
from pipeline.pipeline import save_pipeline_result
from schema.extended_schema_validator import ExtendedValidator

logger = logging.getLogger()
logger.propagate = True
logger.level = logging.INFO
imgui_handler = ImGuiHandler()
logger.addHandler(imgui_handler)


class AppState:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.immvision_params = immvision.ImageParams()
        self.immvision_params.zoom_key = "z"
        self.immvision_params.show_options_panel = False
        self.immvision_params.show_zoom_buttons = False
        self.immvision_params.show_options_button = False
        self.immvision_params.show_image_info = False
        self.immvision_params.show_pixel_info = False
        self.immvision_params.refresh_image = True
        self.immvision_params.can_resize = True

        self.pipelines = defined_pipelines
        self.output_img = None

        self.output_img_stats = None
        self.current_pipeline_result = None
        self.current_pipeline = None
        self.current_pipeline_idx = 0
        self.current_pipeline_step_result_idx = 0
        self.current_pipeline_step_idx = 0

        self.change_current_pipeline()

    def render_ui_pipelines(self):

        self.render_ui_pipeline_list()
        imgui.spacing()
        imgui.spacing()
        self.render_ui_pipeline_config()
        imgui.spacing()
        imgui.spacing()
        self.render_ui_run_pipeline()
    def render_ui_log(self):

        flags: imgui.InputTextFlags = (
                imgui.InputTextFlags_.read_only.value
                | imgui.InputTextFlags_.no_horizontal_scroll.value
        )
        imgui.set_next_item_width(-imgui.FLT_MIN)
        imgui.input_text_multiline('##LogInpLabel', '\n'.join(imgui_handler.buffer), flags=flags)

    def render_ui_pipeline_result(self):
        if self.current_pipeline_result:

            clicked, self.current_pipeline_step_result_idx = imgui.list_box(
                "##PipelineResultListBox",
                self.current_pipeline_step_result_idx,
                [self.current_pipeline.steps[i][0] for i, x in enumerate(self.current_pipeline_result.step_results)]
            )
            if clicked:
                self.prepare_pipeline_result()

            imgui.same_line()

            imgui.begin_group()

            if imgui.button('Save Results as Images'):
                save_pipeline_result(self.current_pipeline, self.current_pipeline_result, 'results/')

            if self.output_img_stats:
                imgui.text_colored(imgui.ImVec4(0.2, 1., 0., 1.), f'Dimensions: {self.output_img_stats["dims"]}')
                imgui.text_colored(imgui.ImVec4(0.2, 1., 0., 1.), f'Mean:')
                imgui.same_line()
                for mean in self.output_img_stats["mean"]:
                    imgui.text_colored(imgui.ImVec4(0.2, 1., 0., 1.), f'{mean:.3f} ')
                    imgui.same_line()
                imgui.new_line()
                imgui.text_colored(imgui.ImVec4(0.2, 1., 0., 1.), f'Entropy:')
                imgui.same_line()
                for entropy in self.output_img_stats["entropy"]:
                    imgui.text_colored(imgui.ImVec4(0.2, 1., 0., 1.), f'{entropy:.3f} ')
                    imgui.same_line()

            imgui.end_group()

            if self.output_img_stats:

                imgui.begin_child('##StatsRegion', size=imgui.ImVec2(imgui.get_content_region_avail()[0], 160),
                                  window_flags=imgui.WindowFlags_.horizontal_scrollbar)

                imgui.label_text('##Histos', 'Histograms')
                imgui.begin_group()

                for channel_histo in self.output_img_stats['histo']:
                    region = imgui.get_content_region_avail()

                    # Necessary for uv + vscode, no empty string
                    imgui.plot_histogram('##hist0', channel_histo[0].astype(np.float32), graph_size=(256, 100))
                    imgui.same_line()

                imgui.end_group()

                imgui.end_child()

            self.render_ui_output()

    def render_ui_pipeline_list(self):
        if self.pipelines:

            imgui.begin_group()

            imgui.text('Available Processing Pipelines')

            full_width = -imgui.FLT_MIN
            item_height = imgui.get_text_line_height_with_spacing()
            items = [x.name for x in self.pipelines]
            if imgui.begin_list_box("##PipelinesListBox", imgui.ImVec2(full_width, 5 * item_height)):
                for n, item in enumerate(items):
                    is_selected = (self.current_pipeline_idx == n)
                    if imgui.selectable(item, is_selected) == (True,True):
                        self.current_pipeline_idx = n
                        self.change_current_pipeline()
                    # Set the default focus to the selected item
                    if is_selected:
                       imgui.set_item_default_focus()
                imgui.end_list_box()

            imgui.end_group()

    def render_ui_pipeline_config(self):

        imgui.text(f'Configuration for {self.pipelines[self.current_pipeline_idx].name}:')

        full_width = -imgui.FLT_MIN
        item_height = imgui.get_text_line_height_with_spacing()
        items = [x[0] for x in self.current_pipeline.steps]
        if imgui.begin_list_box("##StepListBox", imgui.ImVec2(full_width, 5 * item_height)):
            for n, item in enumerate(items):
                is_selected = (self.current_pipeline_step_idx == n)
                if imgui.selectable(item, is_selected) == (True, True):
                    self.current_pipeline_step_idx = n

                # Set the default focus to the selected item
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_list_box()

        imgui.text(f'Configuration for {self.current_pipeline.steps[self.current_pipeline_step_idx][0]}:')

        self.render_ui_step_configuration(self.current_pipeline.steps[self.current_pipeline_step_idx])

    def render_ui_step_configuration(self, pipeline_step):

        step = pipeline_step[1]
        config_schema = step.config_schema()
        config = pipeline_step[2]

        # necessary to add schema properties that are not yet present in the config.
        try:
            ExtendedValidator(schema=config_schema).validate(config)
        except jsonschema.exceptions.ValidationError as ex:
            imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), ex.message)

        self.render_ui_step_configuration_object(step, config)

    def render_ui_step_configuration_object(self, step, step_config):

        def handle_int_type(property_name, property_schema, property_instance, property_parent):
            current = property_instance
            imgui.text(f'{property_name}:')
            imgui.same_line()
            imgui.set_next_item_width(-imgui.FLT_MIN)
            changed, current = imgui.slider_int(f'##{property_name}',
                                                current,
                                                v_min=property_schema['minimum'],
                                                v_max=property_schema['maximum'])
            if changed:
                property_parent[property_name] = current

        def handle_number_type(property_name, property_schema, property_instance, property_parent):
            current = property_instance
            imgui.text(f'{property_name}:')
            imgui.same_line()
            imgui.set_next_item_width(-imgui.FLT_MIN)
            changed, current = imgui.slider_float(f'##{property_name}',
                                                  current,
                                                  v_min=property_schema['minimum'],
                                                  v_max=property_schema['maximum'])
            if changed:
                property_parent[property_name] = current

        def handle_string_type(property_name, property_schema, property_instance, property_parent):
            imgui.text(f'{property_name}:')
            imgui.same_line()
            imgui.set_next_item_width(-imgui.FLT_MIN)
            if 'enum' in property_schema:
                current = property_schema['enum'].index(property_instance)

                clicked, current = imgui.combo(
                    f'##{property_name}', current, property_schema['enum']
                )
                if clicked:
                    property_parent[property_name] = property_schema['enum'][current]
            else:
                current = property_instance
                changed, current = imgui.input_text(f'##{property_name}', current)

                if changed:
                    property_parent[property_name] = current

        def handle_boolean_type(property_name, property_schema, property_instance, property_parent):
            current = property_instance
            imgui.set_next_item_width(-imgui.FLT_MIN)
            imgui.text(f'{property_name}:')
            imgui.same_line()
            clicked, current = imgui.checkbox(f'##{property_name}', current)
            if clicked:
                property_parent[property_name] = current

        def handle_array_type(property_name, property_schema, property_instance, property_parent):
            imgui.text('Cannot handle arrays.')

        def handle_object_type(object_name, object_schema, object_instance, object_parent):
            if object_name:
                imgui.set_next_item_width(-imgui.FLT_MIN)
                if imgui.tree_node(object_name):
                    handle_properties(object_instance, object_schema)
                    imgui.tree_pop()
            else:
                handle_properties(object_instance, object_schema)

        def handle_properties(object_instance, object_schema):
            if 'properties' not in object_schema:
                return
            for property_name, property_schema in object_schema['properties'].items():
                if 'type' not in property_schema:
                    continue

                property_type_handler[property_schema['type']](
                    property_name,
                    property_schema,
                    object_instance[property_name],
                    object_instance);

        property_type_handler = {
            'string': handle_string_type,
            'integer': handle_int_type,
            'array': handle_array_type,
            'number': handle_number_type,
            'boolean': handle_boolean_type,
            'object': handle_object_type
        }

        handle_object_type(None, step.config_schema(), step_config, None)

    def render_ui_run_pipeline(self):

        imgui.begin_group()

        if imgui.button("Run Pipeline"):
            if self.current_pipeline:

                for x in self.current_pipeline.steps:
                    print(x[2])

                self.current_pipeline_result = self.current_pipeline.execute()
                self.prepare_pipeline_result()

        imgui.end_group()

    def render_ui_output(self):

        imgui.begin_child("##ResultRegion")  # , border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)

        if self.output_img is not None:
            self.immvision_params.image_display_size = (int(imgui.get_content_region_avail().x),
                                                        int(imgui.get_content_region_avail().y))
            immvision.image("", self.output_img / 255., self.immvision_params)

        imgui.end_child()

    def change_current_pipeline(self):
        self.current_pipeline = self.pipelines[self.current_pipeline_idx]
        self.current_pipeline_result = None
        self.current_pipeline_step_result_idx = 0
        self.current_pipeline_step_idx = 0

    def prepare_pipeline_result(self):
        if self.current_pipeline_result.successful_execution():
            self.output_img = self.current_pipeline_result.step_results[
                self.current_pipeline_step_result_idx].output_img

            print(f'shape: {self.output_img.shape[1::-1]}')

            self.output_img_stats = compute_img_stats(self.output_img)


def main():

    # Necessary for uv + vscode
    immvision.use_rgb_color_order()

    # Our application state
    app_state = AppState()

    # Hello ImGui params (they hold the settings as well as the Gui callbacks)
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Image Processing Framework"
    runner_params.imgui_window_params.menu_app_title = "IPF"
    runner_params.app_window_params.window_geometry.window_size_state = hello_imgui.WindowSizeState.maximized
    runner_params.app_window_params.borderless = False
    runner_params.app_window_params.borderless_movable = True
    runner_params.app_window_params.borderless_resizable = True
    runner_params.app_window_params.borderless_closable = True

    # Status bar
    runner_params.imgui_window_params.show_status_bar = True
    # runner_params.callbacks.show_status = lambda: app_state.render_status_bar_ui()

    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )

    runner_params.imgui_window_params.enable_viewports = True
    runner_params.docking_params = create_layout(app_state)

    runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    runner_params.ini_filename = "IPF/params.ini"

    hello_imgui.run(runner_params)


def create_docking_splits():
    #    ___________________________________________
    #    |           |                             |
    #    | Pipelines |                             |
    #    | Space     |    MainDockSpace            |
    #    |           |                             |
    #    |           |                             |
    #    |           |                             |
    #    -------------------------------------------
    #    |     LogSpace                            |
    #    -------------------------------------------

    split_log = hello_imgui.DockingSplit()
    split_log.initial_dock = "MainDockSpace"
    split_log.new_dock = "LogSpace"
    split_log.direction = imgui.Dir_.down
    split_log.ratio = 0.20

    split_pipelines = hello_imgui.DockingSplit()
    split_pipelines.initial_dock = "MainDockSpace"
    split_pipelines.new_dock = "PipelinesSpace"
    split_pipelines.direction = imgui.Dir_.left
    split_pipelines.ratio = 0.15

    splits = [split_log, split_pipelines]
    return splits


def create_dockable_windows(app_state):
    pipelines_window = hello_imgui.DockableWindow()
    pipelines_window.can_be_closed = False
    pipelines_window.label = "Pipelines"
    pipelines_window.dock_space_name = "PipelinesSpace"
    pipelines_window.gui_function = lambda: app_state.render_ui_pipelines()

    result_window = hello_imgui.DockableWindow()
    result_window.can_be_closed = False
    result_window.label = "Result"
    result_window.dock_space_name = "MainDockSpace"
    result_window.gui_function = lambda: app_state.render_ui_pipeline_result()

    logs_window = hello_imgui.DockableWindow()
    logs_window.can_be_closed = False
    logs_window.label = "Logs"
    logs_window.dock_space_name = "LogSpace"
    logs_window.gui_function = lambda: app_state.render_ui_log()

    dockable_windows = [
        pipelines_window,
        result_window,
        logs_window
    ]

    return dockable_windows


def create_layout(app_state):
    docking_params = hello_imgui.DockingParams()
    docking_params.docking_splits = create_docking_splits()
    docking_params.dockable_windows = create_dockable_windows(app_state)
    return docking_params


if __name__ == '__main__':
    logging.basicConfig(filename='log.log', filemode='w', level=logging.INFO)
    main()
