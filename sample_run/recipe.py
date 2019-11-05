context.push_event('calibrate_wavelengths', action.args)
context.push_event('remove_emission_line_regions', action.args)
context.push_event('remove_solar_regions', action.args)
context.push_event('correct_telluric_lines', action.args)
if action.args:
    context.push_event('correct_wavelength_dependent_barycentric_velocity', action.args)
    context.push_event('calculate_RV_from_spectrum', action.args)
