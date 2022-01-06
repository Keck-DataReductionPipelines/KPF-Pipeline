KPF Pipeline Configs
====================

As described in :doc:`start`, to run pipeline command  ``kpf``, two input arguments, a ``recipe`` file and a ``config`` configuration file, are
needed. To build the configuration file, KPF Pipeline uses the configuration language that Python ``ConfigPaser`` class implements for Python programs.  
The syntax of the configuration language primarily consists of ``key = value``, ``[section name]``, and ``# comments`` (or  ``; comments``) and the configuration
file consists of sections followed by key/value pairs of options.  

The creation of the key/value pairs, in general, follows the convention as those provided by the ``ConfigParser`` classs.  Specifically for KPF Pipeline, the key consists of alphanumerics and _ except the keywords 
defined in KPF Pipeline (e.g. ``KPFPIPE_TEST_DATA`` for the path of the test data) and the value is in the form of a string, a number, a boolean, or a list. 


For KPF Pipeline, the configuration file needs to include the following three sections. As mentioned in :doc:`example_trivial`, 
there might be no key/value defined under the section but the section header is mandatory.
::

	# KPF Pipeline logging configuration 
	[LOGGER]

	# reference used in the recipe in the form like config.ARGUMENT.<key name>
	[ARGUMENT]

	# the path of the configuration file for the module 
	[MODULE_CONFIGS]


The section ``LOGGER`` contains key/value pairs for the logging configuration for KPF Pipeline, i.e. the second logging source listed in :doc:`logging`.  
The key name and the value options for the key except ``log_path`` are predefined in the logger. Each key gets a default value if it is not set in the section.
::

	[LOGGER]
	start_log = True | False       	  			# if logging or not, default: False
 	log_path = <path of the file>     			# path of KPF Pipeline logging file, default: log 
	log_level = info | debug | error | critical | warning  	# logging level, default: warning
	log_verbose = True | False 				# if printing the log to the terminal, default: True


The section ``ARGUMENT`` contains the key/value pairs referenced in the recipe script. The recipe gets the value of the key in the form of  ``config.ARGUMENT.<key name>``. 
::

	*** configuration file ***
	
	[ARGUMENT]
	data_type = 'NEID'
	input_flat_file_pattern = '/FLAT/stacked_2fiber*.fits'
	data_dir = '/NEIDdata' 
 	:
	obs_list = ['023129', '023815', '024240', '024704', '025129', '025613', '030057', '030724']
	max_result_order = -1
	:


	*** recipe file ***
	
	data_type = config.ARGUMENT.data_type
	data_dir = config.ARGUMENT.data_dir
	input_flat_pattern = KPFPIPE_TEST_DATA + data_dir +  config.ARGUMENT.input_flat_file_pattern
	:	
	obs_list = config.ARGUMENT.obs_list
        :
	max_result_order = config.ARGUMENT.max_result_order


.. note:: 
	Due to the use of Python's ``ConfigParser`` class to parse the configuration file, the keys in sections of 
        the configuration file are case-insensitive and stored in lowercase. 
	Per current pipeline implementation, the recipe should reach the key in lowercase even the key 
        includes uppercase characters. 
	It is recommended to use the lowercase to name the key in the configuration file.  
		
The section ``MODULE_CONFIGS`` contains key/value pairs for the path of the configuration file or others which is the module or/and mission dependent. The following example lists the paths of three configuration files for the modules invoked in the pipeline. 
::

	[MODULE_CONFIGS]
	order_trace = modules/order_trace/configs/default_recipe_neid.cfg
	spectral_extraction = modules/spectral_extraction/configs/default_recipe_neid.cfg
	radial_velocity = modules/radial_velocity/configs/default_recipe_neid.cfg


The path defined in the section ``MODULE_CONFIGS`` can be accessed from the context passed to the module. The following code example shows how the module ``OrderTrace`` retrieves the path of the configuration file with the settings for finding the order trace.  
::
 
	class OrderTrace(KPF0_Primitive)
	    def __init__(self, 
        		action: Action,
        		context: ProcessingContext) -> None:

		:
		self.config = configparser.ConfigParser()
		try:
	            self.config_path = context.config_path['order_trace']  # config path for 'order_trace'
		except:
        	    self.config_path = DEFAULT_CFG_PATH
		
        	self.config.read(self.config_path)			   # parse the config file
		:


For more detail, please refer to the source in :doc:`../primitives/order_trace/primitive_order_trace`.
