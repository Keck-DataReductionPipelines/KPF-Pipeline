# An example pipeline that is used to test the template fitting 
# algorithm module. 
import os
import importlib

from keckdrpframework.pipelines.base_pipeline import Base_pipeline

class TestPipeline(Base_pipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework

    Attributes:
        event_table (dictionary): table of actions known to framework. All primitives must be registered here.
    """
    # Modification: 
    # The event table now contains function handles to the event. It is now 
    # updated dynmaically upon pipeline initialization. 
    

    def __init__(self):
        Base_pipeline.__init__(self)

    def start(self, config):
        '''
        Initialize the pipeline, which can't be done in __init__ since 
        it requires input argument, which is prohibited by the framework
        '''
        # By this point we should have the following:
        # - configuration class of ConfigParser
        self.config = config
        self.name = self.config.get('pipeline_name')
        self.mod_search_path = self.config.get('mod_search_path')

        # Populate event table with handles. 
        # we are assuming that the keys are the exact name of the module
        # Module can be members of this pipeline class, or seperate classes 
        # implemented anywhere in the module_search_path folder
        self.event_table = {
            "evaluate_recipe": self.evaluate_recipe, 
            "exit": self.exit_loop
        }
        # populate the event table:
        # First loop through all module folders
        for mod_folder in os.listdir(self.mod_search_path):
            # loop through all files in this folder. 
            # If it's a .py that satisfies the naming convention, try to import it
            relative_path = '{}/{}'.format(self.mod_search_path, mod_folder)
            if os.path.isdir(relative_path):
                for modpy in os.listdir(relative_path):
                    if self.check_name(modpy):
                        print(modpy)
                        # This file is a valid KPF module 
                        mod_name = modpy.split('.')[0]
                        print(mod_name)
                        import_path = '{}.{}.{}'.format('modules', mod_folder, mod_name)
                        module = importlib.import_module(import_path)
                        # the name of the primitvie class is the same as the file name
                        primitive_handle = getattr(module, mod_name)
                        # Populate event table
                        self.event_table[mod_name] = primitive_handle
        print(self.event_table)
            
    def check_name(self, mod: str) -> bool:
        success = mod.endswith('.py')
        success &= mod.startswith('KPFModule_')
        return success



    def evaluate_recipe(self, action, context):
        """
        Evaluates the recipe file (Python file) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        recipe_file = action.args.recipe
        context.logger.info('Executing recipe file {}'.format(recipe_file))
        f = open(recipe_file)
        fstr = f.read()
        exec(fstr)
        f.close()

    def exit_loop(self, action, context):
        """
        Force the Keck DRP Framework to exit the infinite loop

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        context.logger.info("Goodbye")
        os._exit(0)

    def get_pre_action(self, action_name):
        # Dummpy place holder
        def f(action, context):
            return True
        return f

    def get_action(self, action_name):
        return self.event_table[action_name]
        

    def get_post_action(self, action_name):
        # Dummpy place holder
        def f(action, context):
            return True
        return f
    
    def event_to_action(self, event, context):
        return (event.name, None, None)