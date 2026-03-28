import bpy


def connectome_geometry_1_node_group():
    """Initialize Connectome geometry node group"""
    connectome_geometry_1 = bpy.data.node_groups.new(type='GeometryNodeTree', name="Connectome geometry")

    connectome_geometry_1.color_tag = 'CONVERTER'
    connectome_geometry_1.description = ""
    connectome_geometry_1.default_group_node_width = 140
    connectome_geometry_1.is_modifier = True
    connectome_geometry_1.is_tool = True
    connectome_geometry_1.is_mode_object = False
    connectome_geometry_1.is_mode_edit = False
    connectome_geometry_1.is_mode_sculpt = False
    connectome_geometry_1.is_type_curve = False
    connectome_geometry_1.is_type_mesh = False
    connectome_geometry_1.is_type_point_cloud = False

    # connectome_geometry_1 interface

    # Socket Instances
    instances_socket = connectome_geometry_1.interface.new_socket(name="Instances", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    instances_socket.attribute_domain = 'POINT'

    # Socket Geometry
    geometry_socket = connectome_geometry_1.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    geometry_socket.attribute_domain = 'POINT'

    # Socket Scale
    scale_socket = connectome_geometry_1.interface.new_socket(name="Scale", in_out='INPUT', socket_type='NodeSocketFloat')
    scale_socket.default_value = 9.999999747378752e-05
    scale_socket.min_value = -3.4028234663852886e+38
    scale_socket.max_value = 3.4028234663852886e+38
    scale_socket.subtype = 'NONE'
    scale_socket.attribute_domain = 'POINT'

    # Socket PointRadius
    pointradius_socket = connectome_geometry_1.interface.new_socket(name="PointRadius", in_out='INPUT', socket_type='NodeSocketFloat')
    pointradius_socket.default_value = 0.10000000149011612
    pointradius_socket.min_value = 0.0
    pointradius_socket.max_value = 3.4028234663852886e+38
    pointradius_socket.subtype = 'DISTANCE'
    pointradius_socket.attribute_domain = 'POINT'

    # Socket NeuronSubdivisions
    neuronsubdivisions_socket = connectome_geometry_1.interface.new_socket(name="NeuronSubdivisions", in_out='INPUT', socket_type='NodeSocketInt')
    neuronsubdivisions_socket.default_value = 1
    neuronsubdivisions_socket.min_value = 1
    neuronsubdivisions_socket.max_value = 7
    neuronsubdivisions_socket.subtype = 'NONE'
    neuronsubdivisions_socket.attribute_domain = 'POINT'

    # Initialize connectome_geometry_1 nodes

    # Node Group Output
    group_output = connectome_geometry_1.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # Node Group Input
    group_input = connectome_geometry_1.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # Node Instance on Points
    instance_on_points = connectome_geometry_1.nodes.new("GeometryNodeInstanceOnPoints")
    instance_on_points.name = "Instance on Points"
    # Selection
    instance_on_points.inputs[1].default_value = True
    # Pick Instance
    instance_on_points.inputs[3].default_value = False
    # Instance Index
    instance_on_points.inputs[4].default_value = 0
    # Rotation
    instance_on_points.inputs[5].default_value = (0.0, 0.0, 0.0)
    # Scale
    instance_on_points.inputs[6].default_value = (1.0, 1.0, 1.0)

    # Node Transform Geometry
    transform_geometry = connectome_geometry_1.nodes.new("GeometryNodeTransform")
    transform_geometry.name = "Transform Geometry"
    transform_geometry.mode = 'COMPONENTS'
    # Translation
    transform_geometry.inputs[1].default_value = (0.0, 0.0, 0.0)
    # Rotation
    transform_geometry.inputs[2].default_value = (-1.5707963705062866, 0.0, 0.0)

    # Node Ico Sphere
    ico_sphere = connectome_geometry_1.nodes.new("GeometryNodeMeshIcoSphere")
    ico_sphere.name = "Ico Sphere"

    # Node Set Shade Smooth
    set_shade_smooth = connectome_geometry_1.nodes.new("GeometryNodeSetShadeSmooth")
    set_shade_smooth.name = "Set Shade Smooth"
    set_shade_smooth.domain = 'FACE'
    # Selection
    set_shade_smooth.inputs[1].default_value = True
    # Shade Smooth
    set_shade_smooth.inputs[2].default_value = True

    # Node Combine XYZ
    combine_xyz = connectome_geometry_1.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.name = "Combine XYZ"

    # Node Set Material
    set_material = connectome_geometry_1.nodes.new("GeometryNodeSetMaterial")
    set_material.name = "Set Material"
    # Selection
    set_material.inputs[1].default_value = True
    if "Neuron" in bpy.data.materials:
        set_material.inputs[2].default_value = bpy.data.materials["Neuron"]

    # Node Viewer
    viewer = connectome_geometry_1.nodes.new("GeometryNodeViewer")
    viewer.name = "Viewer"
    viewer.data_type = 'FLOAT'
    viewer.domain = 'POINT'
    # Value
    viewer.inputs[1].default_value = 0.0

    # Node Viewer.001
    viewer_001 = connectome_geometry_1.nodes.new("GeometryNodeViewer")
    viewer_001.name = "Viewer.001"
    viewer_001.data_type = 'FLOAT'
    viewer_001.domain = 'POINT'
    # Value
    viewer_001.inputs[1].default_value = 0.0

    # Node Transform Geometry.001
    transform_geometry_001 = connectome_geometry_1.nodes.new("GeometryNodeTransform")
    transform_geometry_001.name = "Transform Geometry.001"
    transform_geometry_001.mode = 'COMPONENTS'
    # Rotation
    transform_geometry_001.inputs[2].default_value = (0.0, 0.0, 0.0)
    # Scale
    transform_geometry_001.inputs[3].default_value = (1.0, 1.0, 1.0)

    # Node Vector Math
    vector_math = connectome_geometry_1.nodes.new("ShaderNodeVectorMath")
    vector_math.name = "Vector Math"
    vector_math.operation = 'SCALE'
    # Vector
    vector_math.inputs[0].default_value = (-5.199999809265137, -12.499999046325684, 0.0)

    # Node Math
    math = connectome_geometry_1.nodes.new("ShaderNodeMath")
    math.name = "Math"
    math.operation = 'DIVIDE'
    math.use_clamp = False
    # Value
    math.inputs[0].default_value = 1.0

    # Set locations
    connectome_geometry_1.nodes["Group Output"].location = (1341.7647705078125, -8.857583999633789)
    connectome_geometry_1.nodes["Group Input"].location = (-813.5371704101562, 63.158226013183594)
    connectome_geometry_1.nodes["Instance on Points"].location = (588.541748046875, 19.52813720703125)
    connectome_geometry_1.nodes["Transform Geometry"].location = (180.6524200439453, 142.9449920654297)
    connectome_geometry_1.nodes["Ico Sphere"].location = (166.0368194580078, -240.44261169433594)
    connectome_geometry_1.nodes["Set Shade Smooth"].location = (1004.7981567382812, 5.518120765686035)
    connectome_geometry_1.nodes["Combine XYZ"].location = (-19.36151885986328, -152.76089477539062)
    connectome_geometry_1.nodes["Set Material"].location = (803.0000610351562, -49.10784149169922)
    connectome_geometry_1.nodes["Viewer"].location = (982.5204467773438, 167.2443084716797)
    connectome_geometry_1.nodes["Viewer.001"].location = (19.02056884765625, 210.93014526367188)
    connectome_geometry_1.nodes["Transform Geometry.001"].location = (-190.6190185546875, 129.56527709960938)
    connectome_geometry_1.nodes["Vector Math"].location = (-392.4460144042969, 325.50347900390625)
    connectome_geometry_1.nodes["Math"].location = (-619.75341796875, 228.94949340820312)

    # Set dimensions
    connectome_geometry_1.nodes["Group Output"].width  = 140.0
    connectome_geometry_1.nodes["Group Output"].height = 100.0

    connectome_geometry_1.nodes["Group Input"].width  = 140.0
    connectome_geometry_1.nodes["Group Input"].height = 100.0

    connectome_geometry_1.nodes["Instance on Points"].width  = 140.0
    connectome_geometry_1.nodes["Instance on Points"].height = 100.0

    connectome_geometry_1.nodes["Transform Geometry"].width  = 140.0
    connectome_geometry_1.nodes["Transform Geometry"].height = 100.0

    connectome_geometry_1.nodes["Ico Sphere"].width  = 140.0
    connectome_geometry_1.nodes["Ico Sphere"].height = 100.0

    connectome_geometry_1.nodes["Set Shade Smooth"].width  = 140.0
    connectome_geometry_1.nodes["Set Shade Smooth"].height = 100.0

    connectome_geometry_1.nodes["Combine XYZ"].width  = 140.0
    connectome_geometry_1.nodes["Combine XYZ"].height = 100.0

    connectome_geometry_1.nodes["Set Material"].width  = 140.0
    connectome_geometry_1.nodes["Set Material"].height = 100.0

    connectome_geometry_1.nodes["Viewer"].width  = 140.0
    connectome_geometry_1.nodes["Viewer"].height = 100.0

    connectome_geometry_1.nodes["Viewer.001"].width  = 140.0
    connectome_geometry_1.nodes["Viewer.001"].height = 100.0

    connectome_geometry_1.nodes["Transform Geometry.001"].width  = 140.0
    connectome_geometry_1.nodes["Transform Geometry.001"].height = 100.0

    connectome_geometry_1.nodes["Vector Math"].width  = 140.0
    connectome_geometry_1.nodes["Vector Math"].height = 100.0

    connectome_geometry_1.nodes["Math"].width  = 140.0
    connectome_geometry_1.nodes["Math"].height = 100.0


    # Initialize connectome_geometry_1 links

    # ico_sphere.Mesh -> instance_on_points.Instance
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Ico Sphere"].outputs[0],
        connectome_geometry_1.nodes["Instance on Points"].inputs[2]
    )
    # group_input.Scale -> combine_xyz.X
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[1],
        connectome_geometry_1.nodes["Combine XYZ"].inputs[0]
    )
    # group_input.Scale -> combine_xyz.Y
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[1],
        connectome_geometry_1.nodes["Combine XYZ"].inputs[1]
    )
    # group_input.Scale -> combine_xyz.Z
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[1],
        connectome_geometry_1.nodes["Combine XYZ"].inputs[2]
    )
    # combine_xyz.Vector -> transform_geometry.Scale
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Combine XYZ"].outputs[0],
        connectome_geometry_1.nodes["Transform Geometry"].inputs[3]
    )
    # set_material.Geometry -> viewer.Geometry
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Set Material"].outputs[0],
        connectome_geometry_1.nodes["Viewer"].inputs[0]
    )
    # set_shade_smooth.Geometry -> group_output.Instances
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Set Shade Smooth"].outputs[0],
        connectome_geometry_1.nodes["Group Output"].inputs[0]
    )
    # transform_geometry_001.Geometry -> transform_geometry.Geometry
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Transform Geometry.001"].outputs[0],
        connectome_geometry_1.nodes["Transform Geometry"].inputs[0]
    )
    # transform_geometry.Geometry -> instance_on_points.Points
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Transform Geometry"].outputs[0],
        connectome_geometry_1.nodes["Instance on Points"].inputs[0]
    )
    # instance_on_points.Instances -> set_material.Geometry
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Instance on Points"].outputs[0],
        connectome_geometry_1.nodes["Set Material"].inputs[0]
    )
    # set_material.Geometry -> set_shade_smooth.Geometry
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Set Material"].outputs[0],
        connectome_geometry_1.nodes["Set Shade Smooth"].inputs[0]
    )
    # group_input.Geometry -> viewer_001.Geometry
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[0],
        connectome_geometry_1.nodes["Viewer.001"].inputs[0]
    )
    # group_input.PointRadius -> ico_sphere.Radius
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[2],
        connectome_geometry_1.nodes["Ico Sphere"].inputs[0]
    )
    # group_input.NeuronSubdivisions -> ico_sphere.Subdivisions
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[3],
        connectome_geometry_1.nodes["Ico Sphere"].inputs[1]
    )
    # group_input.Geometry -> transform_geometry_001.Geometry
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[0],
        connectome_geometry_1.nodes["Transform Geometry.001"].inputs[0]
    )
    # vector_math.Vector -> transform_geometry_001.Translation
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Vector Math"].outputs[0],
        connectome_geometry_1.nodes["Transform Geometry.001"].inputs[1]
    )
    # math.Value -> vector_math.Scale
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Math"].outputs[0],
        connectome_geometry_1.nodes["Vector Math"].inputs[3]
    )
    # group_input.Scale -> math.Value
    connectome_geometry_1.links.new(
        connectome_geometry_1.nodes["Group Input"].outputs[1],
        connectome_geometry_1.nodes["Math"].inputs[1]
    )

    return connectome_geometry_1

neuron = bpy.data.materials.get("Neuron")
if neuron  is None:
    neuron = bpy.data.materials.new(name = "Neuron")
    neuron.alpha_threshold = 0.5
    neuron.line_priority = 0
    neuron.max_vertex_displacement = 0.0
    neuron.metallic = 0.0
    neuron.paint_active_slot = 0
    neuron.paint_clone_slot = 0
    neuron.pass_index = 0
    neuron.refraction_depth = 0.0
    neuron.roughness = 0.4000000059604645
    neuron.show_transparent_back = True
    neuron.specular_intensity = 0.5
    neuron.use_backface_culling = False
    neuron.use_backface_culling_lightprobe_volume = True
    neuron.use_backface_culling_shadow = False
    neuron.use_preview_world = False
    neuron.use_raytrace_refraction = False
    neuron.use_screen_refraction = False
    neuron.use_sss_translucency = False
    neuron.use_thickness_from_shadow = False
    neuron.use_transparency_overlap = True
    neuron.use_transparent_shadow = True
    neuron.blend_method = 'HASHED'
    neuron.displacement_method = 'BUMP'
    neuron.preview_render_type = 'SPHERE'
    neuron.surface_render_method = 'DITHERED'
    neuron.thickness_mode = 'SPHERE'
    neuron.volume_intersection_method = 'FAST'
    neuron.specular_color = (1.0, 1.0, 1.0)
    neuron.diffuse_color = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
    neuron.line_color = (0.0, 0.0, 0.0, 0.0)
if bpy.app.version < (5, 0, 0):
    neuron.use_nodes = True



def shader_nodetree_node_group():
    """Initialize Shader Nodetree node group"""
    shader_nodetree = neuron.node_tree

    # Start with a clean node tree
    for node in shader_nodetree.nodes:
        shader_nodetree.nodes.remove(node)
    shader_nodetree.color_tag = 'NONE'
    shader_nodetree.description = ""
    shader_nodetree.default_group_node_width = 140
    # Initialize shader_nodetree nodes

    # Node Material Output
    material_output = shader_nodetree.nodes.new("ShaderNodeOutputMaterial")
    material_output.name = "Material Output"
    material_output.is_active_output = True
    material_output.target = 'ALL'
    # Displacement
    material_output.inputs[2].default_value = (0.0, 0.0, 0.0)
    # Thickness
    material_output.inputs[3].default_value = 0.0

    # Node Principled BSDF
    principled_bsdf = shader_nodetree.nodes.new("ShaderNodeBsdfPrincipled")
    principled_bsdf.name = "Principled BSDF"
    principled_bsdf.distribution = 'MULTI_GGX'
    principled_bsdf.subsurface_method = 'RANDOM_WALK'
    # Base Color
    principled_bsdf.inputs[0].default_value = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
    # Metallic
    principled_bsdf.inputs[1].default_value = 0.0
    # Roughness
    principled_bsdf.inputs[2].default_value = 0.5
    # IOR
    principled_bsdf.inputs[3].default_value = 1.5
    # Alpha
    principled_bsdf.inputs[4].default_value = 1.0
    # Normal
    principled_bsdf.inputs[5].default_value = (0.0, 0.0, 0.0)
    # Diffuse Roughness
    principled_bsdf.inputs[7].default_value = 0.0
    # Subsurface Weight
    principled_bsdf.inputs[8].default_value = 0.0
    # Subsurface Radius
    principled_bsdf.inputs[9].default_value = (1.0, 0.20000000298023224, 0.10000000149011612)
    # Subsurface Scale
    principled_bsdf.inputs[10].default_value = 0.05000000074505806
    # Subsurface Anisotropy
    principled_bsdf.inputs[12].default_value = 0.0
    # Specular IOR Level
    principled_bsdf.inputs[13].default_value = 0.5
    # Specular Tint
    principled_bsdf.inputs[14].default_value = (1.0, 1.0, 1.0, 1.0)
    # Anisotropic
    principled_bsdf.inputs[15].default_value = 0.0
    # Anisotropic Rotation
    principled_bsdf.inputs[16].default_value = 0.0
    # Tangent
    principled_bsdf.inputs[17].default_value = (0.0, 0.0, 0.0)
    # Transmission Weight
    principled_bsdf.inputs[18].default_value = 0.0
    # Coat Weight
    principled_bsdf.inputs[19].default_value = 0.0
    # Coat Roughness
    principled_bsdf.inputs[20].default_value = 0.029999999329447746
    # Coat IOR
    principled_bsdf.inputs[21].default_value = 1.5
    # Coat Tint
    principled_bsdf.inputs[22].default_value = (1.0, 1.0, 1.0, 1.0)
    # Coat Normal
    principled_bsdf.inputs[23].default_value = (0.0, 0.0, 0.0)
    # Sheen Weight
    principled_bsdf.inputs[24].default_value = 0.0
    # Sheen Roughness
    principled_bsdf.inputs[25].default_value = 0.5
    # Sheen Tint
    principled_bsdf.inputs[26].default_value = (1.0, 1.0, 1.0, 1.0)
    # Thin Film Thickness
    principled_bsdf.inputs[29].default_value = 0.0
    # Thin Film IOR
    principled_bsdf.inputs[30].default_value = 1.3300000429153442

    # Node Attribute
    attribute = shader_nodetree.nodes.new("ShaderNodeAttribute")
    attribute.name = "Attribute"
    attribute.attribute_name = "activity"
    attribute.attribute_type = 'INSTANCER'

    # Node Attribute.001
    attribute_001 = shader_nodetree.nodes.new("ShaderNodeAttribute")
    attribute_001.name = "Attribute.001"
    attribute_001.attribute_name = "activity_max"
    attribute_001.attribute_type = 'INSTANCER'

    # Node Attribute.002
    attribute_002 = shader_nodetree.nodes.new("ShaderNodeAttribute")
    attribute_002.name = "Attribute.002"
    attribute_002.attribute_name = "activity_min"
    attribute_002.attribute_type = 'INSTANCER'

    # Node Math
    math = shader_nodetree.nodes.new("ShaderNodeMath")
    math.name = "Math"
    math.operation = 'SUBTRACT'
    math.use_clamp = False

    # Node Math.001
    math_001 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_001.name = "Math.001"
    math_001.operation = 'SUBTRACT'
    math_001.use_clamp = False

    # Node Math.002
    math_002 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_002.name = "Math.002"
    math_002.operation = 'DIVIDE'
    math_002.use_clamp = False

    # Node Color Ramp
    color_ramp = shader_nodetree.nodes.new("ShaderNodeValToRGB")
    color_ramp.name = "Color Ramp"
    color_ramp.color_ramp.color_mode = 'RGB'
    color_ramp.color_ramp.hue_interpolation = 'NEAR'
    color_ramp.color_ramp.interpolation = 'LINEAR'

    # Initialize color ramp elements
    color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
    color_ramp_cre_0 = color_ramp.color_ramp.elements[0]
    color_ramp_cre_0.position = 0.0
    color_ramp_cre_0.alpha = 1.0
    color_ramp_cre_0.color = (1.0, 0.0, 0.0, 1.0)

    color_ramp_cre_1 = color_ramp.color_ramp.elements.new(0.5)
    color_ramp_cre_1.alpha = 1.0
    color_ramp_cre_1.color = (0.0, 0.0, 0.0, 1.0)

    color_ramp_cre_2 = color_ramp.color_ramp.elements.new(1.0)
    color_ramp_cre_2.alpha = 1.0
    color_ramp_cre_2.color = (1.0, 0.9822603464126587, 0.8512240648269653, 1.0)


    # Node Float Curve
    float_curve = shader_nodetree.nodes.new("ShaderNodeFloatCurve")
    float_curve.name = "Float Curve"
    # Mapping settings
    float_curve.mapping.extend = 'EXTRAPOLATED'
    float_curve.mapping.tone = 'STANDARD'
    float_curve.mapping.black_level = (0.0, 0.0, 0.0)
    float_curve.mapping.white_level = (1.0, 1.0, 1.0)
    float_curve.mapping.clip_min_x = 0.0
    float_curve.mapping.clip_min_y = 0.0
    float_curve.mapping.clip_max_x = 1.0
    float_curve.mapping.clip_max_y = 1.0
    float_curve.mapping.use_clip = True
    # Curve 0
    float_curve_curve_0 = float_curve.mapping.curves[0]
    float_curve_curve_0_point_0 = float_curve_curve_0.points[0]
    float_curve_curve_0_point_0.location = (0.0, 0.2500000298023224)
    float_curve_curve_0_point_0.handle_type = 'AUTO'
    float_curve_curve_0_point_1 = float_curve_curve_0.points[1]
    float_curve_curve_0_point_1.location = (0.12727287411689758, 0.10625000298023224)
    float_curve_curve_0_point_1.handle_type = 'AUTO'
    float_curve_curve_0_point_2 = float_curve_curve_0.points.new(0.4681818187236786, 0.0)
    float_curve_curve_0_point_2.handle_type = 'AUTO'
    float_curve_curve_0_point_3 = float_curve_curve_0.points.new(0.6227272748947144, 0.06875015795230865)
    float_curve_curve_0_point_3.handle_type = 'AUTO'
    float_curve_curve_0_point_4 = float_curve_curve_0.points.new(0.8227271437644958, 0.14374993741512299)
    float_curve_curve_0_point_4.handle_type = 'AUTO'
    float_curve_curve_0_point_5 = float_curve_curve_0.points.new(1.0, 1.0)
    float_curve_curve_0_point_5.handle_type = 'AUTO'
    # Update curve after changes
    float_curve.mapping.update()
    # Factor
    float_curve.inputs[0].default_value = 1.0

    # Node Math.004
    math_004 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_004.name = "Math.004"
    math_004.operation = 'MULTIPLY'
    math_004.use_clamp = False
    # Value_001
    math_004.inputs[1].default_value = 1.0

    # Set locations
    shader_nodetree.nodes["Material Output"].location = (1523.8642578125, 73.83457946777344)
    shader_nodetree.nodes["Principled BSDF"].location = (1003.9519653320312, 101.11194610595703)
    shader_nodetree.nodes["Attribute"].location = (-420.9560546875, 187.739501953125)
    shader_nodetree.nodes["Attribute.001"].location = (-470.3641357421875, -347.6534423828125)
    shader_nodetree.nodes["Attribute.002"].location = (-472.1951904296875, -131.62142944335938)
    shader_nodetree.nodes["Math"].location = (-202.846923828125, 15.34765625)
    shader_nodetree.nodes["Math.001"].location = (-193.2685546875, -176.74488830566406)
    shader_nodetree.nodes["Math.002"].location = (59.873382568359375, -32.628173828125)
    shader_nodetree.nodes["Color Ramp"].location = (253.34432983398438, 64.9286880493164)
    shader_nodetree.nodes["Float Curve"].location = (506.4710998535156, -189.5118865966797)
    shader_nodetree.nodes["Math.004"].location = (824.180908203125, -238.27783203125)

    # Set dimensions
    shader_nodetree.nodes["Material Output"].width  = 140.0
    shader_nodetree.nodes["Material Output"].height = 100.0

    shader_nodetree.nodes["Principled BSDF"].width  = 240.0
    shader_nodetree.nodes["Principled BSDF"].height = 100.0

    shader_nodetree.nodes["Attribute"].width  = 140.0
    shader_nodetree.nodes["Attribute"].height = 100.0

    shader_nodetree.nodes["Attribute.001"].width  = 140.0
    shader_nodetree.nodes["Attribute.001"].height = 100.0

    shader_nodetree.nodes["Attribute.002"].width  = 140.0
    shader_nodetree.nodes["Attribute.002"].height = 100.0

    shader_nodetree.nodes["Math"].width  = 140.0
    shader_nodetree.nodes["Math"].height = 100.0

    shader_nodetree.nodes["Math.001"].width  = 140.0
    shader_nodetree.nodes["Math.001"].height = 100.0

    shader_nodetree.nodes["Math.002"].width  = 140.0
    shader_nodetree.nodes["Math.002"].height = 100.0

    shader_nodetree.nodes["Color Ramp"].width  = 240.0
    shader_nodetree.nodes["Color Ramp"].height = 100.0

    shader_nodetree.nodes["Float Curve"].width  = 240.0
    shader_nodetree.nodes["Float Curve"].height = 100.0

    shader_nodetree.nodes["Math.004"].width  = 140.0
    shader_nodetree.nodes["Math.004"].height = 100.0


    # Initialize shader_nodetree links

    # attribute_002.Fac -> math.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute.002"].outputs[2],
        shader_nodetree.nodes["Math"].inputs[1]
    )
    # math_002.Value -> color_ramp.Fac
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.002"].outputs[0],
        shader_nodetree.nodes["Color Ramp"].inputs[0]
    )
    # math_004.Value -> principled_bsdf.Emission Strength
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.004"].outputs[0],
        shader_nodetree.nodes["Principled BSDF"].inputs[28]
    )
    # attribute_001.Fac -> math_001.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute.001"].outputs[2],
        shader_nodetree.nodes["Math.001"].inputs[0]
    )
    # math_001.Value -> math_002.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.001"].outputs[0],
        shader_nodetree.nodes["Math.002"].inputs[1]
    )
    # math_002.Value -> float_curve.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.002"].outputs[0],
        shader_nodetree.nodes["Float Curve"].inputs[1]
    )
    # math.Value -> math_002.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math"].outputs[0],
        shader_nodetree.nodes["Math.002"].inputs[0]
    )
    # float_curve.Value -> math_004.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Float Curve"].outputs[0],
        shader_nodetree.nodes["Math.004"].inputs[0]
    )
    # color_ramp.Color -> principled_bsdf.Emission Color
    shader_nodetree.links.new(
        shader_nodetree.nodes["Color Ramp"].outputs[0],
        shader_nodetree.nodes["Principled BSDF"].inputs[27]
    )
    # attribute_002.Fac -> math_001.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute.002"].outputs[2],
        shader_nodetree.nodes["Math.001"].inputs[1]
    )
    # attribute.Fac -> math.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute"].outputs[2],
        shader_nodetree.nodes["Math"].inputs[0]
    )
    # principled_bsdf.BSDF -> material_output.Surface
    shader_nodetree.links.new(
        shader_nodetree.nodes["Principled BSDF"].outputs[0],
        shader_nodetree.nodes["Material Output"].inputs[0]
    )

    return shader_nodetree
    


# ----------------------------------------------------

import numpy as np
import os
from collections import namedtuple


connectome_context = namedtuple('connectome_context', ['activity_matrix', 'coords', 'activity_path', 'coords_path', 'neurons_ids', 'activity_max', 'activity_min'])


def load_data(ACTIVITY_PATH, POSITION_PATH):
    raw_coords = np.load(POSITION_PATH)
    raw_activity = np.load(ACTIVITY_PATH)
    neuron_ids = sorted([int(k) for k in raw_coords.keys() if k != '/t']) # можно не сортировать, но мне спокойней если отсортировать
    time_points = raw_activity['/t' if '/t' in raw_activity.keys() else f"/voltages/{neuron_ids[0]}"]
    num_frames = len(time_points)
    num_neurons = len(neuron_ids)
    activity_matrix = np.zeros((num_frames, num_neurons), dtype=np.float32)
    activity_max = np.zeros((num_neurons,), dtype=np.float32)
    activity_min = np.zeros((num_neurons,), dtype=np.float32)
    for i, nid in enumerate(neuron_ids):
        voltage_key = f"/voltages/{nid}"
        if voltage_key in raw_activity:
            activity_matrix[:, i] = raw_activity[voltage_key][:num_frames]
            activity_max[i] = raw_activity[voltage_key].max()
            activity_min[i] = raw_activity[voltage_key].min()
        else:
            print(f"no {voltage_key} in activity")
    coords = np.array([raw_coords[str(nid)] for nid in neuron_ids])

    return connectome_context(
        activity_matrix=activity_matrix,
        coords = coords,
        activity_path = ACTIVITY_PATH,
        coords_path = POSITION_PATH,
        neurons_ids = neuron_ids,
        activity_max = activity_max,
        activity_min = activity_min
    )


def create_connectome_mesh(ctx:connectome_context, name = None):
    if name is None:
        name = os.path.splitext(os.path.basename(ctx.activity_path))[0]

    mesh = bpy.data.meshes.new(f"N_{name}Mesh")
    obj = bpy.data.objects.new(f"N_{name}", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(ctx.coords, [], [])

    # attrs
    if "activity" not in mesh.attributes:
        mesh.attributes.new(name="activity", type='FLOAT', domain='POINT')
    
    if "neurons_ids" not in mesh.attributes:
        mesh.attributes.new(name="neurons_ids", type='INT', domain='POINT')
    obj.data.attributes["neurons_ids"].data.foreach_set("value", ctx.neurons_ids)
    
    if "activity_max" not in mesh.attributes:
        mesh.attributes.new(name="activity_max", type='FLOAT', domain='POINT')
    obj.data.attributes["activity_max"].data.foreach_set("value", ctx.activity_max)

    if "activity_min" not in mesh.attributes:
        mesh.attributes.new(name="activity_min", type='FLOAT', domain='POINT')
    obj.data.attributes["activity_min"].data.foreach_set("value", ctx.activity_min)

    obj["activity_path"] = ctx.activity_path
    obj["coords_path"] = ctx.coords_path
    return obj


def create_handler(obj, ctx:connectome_context):
    def update_neurons(scene):
        frame = scene.frame_current % ctx.activity_matrix.shape[0]

        if "activity" in obj.data.attributes:
            obj.data.attributes["activity"].data.foreach_set("value", ctx.activity_matrix[frame])
            obj.data.update()
        else:
            raise AttributeError()
    return update_neurons


def apply_handlers():
    scene = bpy.context.scene
    bpy.app.handlers.frame_change_pre.clear()
    for obj in scene.objects:
        activity_path = obj.get("activity_path", None)
        if activity_path is not None:
            ctx = load_data(activity_path, obj['coords_path'])
            handler = create_handler(obj, ctx)
            bpy.app.handlers.frame_change_pre.append(handler)

class ConnectomePanel(bpy.types.Panel):
    bl_label = "Connectome"
    bl_idname = "object.connectome_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Connectome'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Рисуем поле ввода пути
        # 'path_to_folder' — это имя свойства, которое мы создадим ниже
        col = layout.column(align=True)
        col.prop(scene, "path_to_activity", text="Activity")
        col.prop(scene, "path_to_coords", text="Coords")
        
        # Кнопка для запуска какого-либо действия с этим путем
        layout.operator("object.process_paths")

class ProcessPaths(bpy.types.Operator):
    bl_idname = "object.process_paths"
    bl_label = "Create object"

    def execute(self, context):
        path_to_activity = context.scene.path_to_activity
        path_to_coords = context.scene.path_to_coords
        ctx = load_data(path_to_activity, path_to_coords)
        obj = create_connectome_mesh(ctx)
        tree = bpy.data.node_groups.get("Connectome geometry", None)
        assert tree is not None
        apply_gn_to_object(obj, tree)

        apply_handlers()
        return {'FINISHED'}


def apply_gn_to_object(obj, tree):
    # Ищем, есть ли уже модификатор Geometry Nodes
    mod = None
    for m in obj.modifiers:
        if m.type == 'NODES' and m.name == "NeuroNodes":
            mod = m
            break

    # Если не нашли — создаем
    if mod is None:
        mod = obj.modifiers.new(name="Connectome geometry", type='NODES')
    
    # Назначаем дерево
    mod.node_group = tree


def unregister():
    # 1. Удаляем хендлеры, чтобы скрипт не пытался выполниться после выключения
    bpy.app.handlers.frame_change_pre.clear()

    # 2. Удаляем свойства (важно, чтобы не засорять API)
    del bpy.types.Scene.path_to_activity
    del bpy.types.Scene.path_to_coords

    # 3. Удаляем классы
    bpy.utils.unregister_class(ConnectomePanel)
    bpy.utils.unregister_class(ProcessPaths)

def register():
    # Создаем динамическое свойство в объекте Scene
    bpy.types.Scene.path_to_activity = bpy.props.StringProperty(
        name="Activity",
        description="path to activity",
        default= "",
        maxlen=1024,
        subtype='FILE_PATH'
    )

    bpy.types.Scene.path_to_coords = bpy.props.StringProperty(
        name="Positions",
        description="path to positions",
        default="",
        maxlen=1024,
        subtype='FILE_PATH'
    )

    bpy.utils.register_class(ConnectomePanel)
    bpy.utils.register_class(ProcessPaths)

if __name__ == "__main__":

    connectome_geometry = bpy.data.node_groups.get("Connectome geometry", None)
    if connectome_geometry is None:
        connectome_geometry = connectome_geometry_1_node_group()
    shader_nodetree = shader_nodetree_node_group()

    # Проверка: есть ли материал в файле
    
    register()
    apply_handlers()