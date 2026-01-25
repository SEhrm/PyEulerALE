project('py_euler_ale', 'c', 'fortran',
  meson_version: '>=0.64.0',
  default_options: [
    'optimization=3',
  ]
)
add_project_arguments('-Wall', '-Wextra', '-pedantic', '-Werror', '--std=f2008', '-march=native', language: 'fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()
incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

f2py_mod_name = 'euler_ale'
fortran_source = 'src/spatial_discretization.f90'
f2py_mod_source = custom_target(f2py_mod_name+'module.c',
  input : [fortran_source],
  output : [f2py_mod_name+'module.c', f2py_mod_name+'-f2pywrappers2.f90'],  # hard coded in f2py
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', f2py_mod_name, '--lower']
)
py.extension_module(f2py_mod_name,
  [fortran_source, f2py_mod_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: include_directories(incdir_numpy, incdir_f2py),
  dependencies: py_dep,
  install: true,
  subdir: 'py_euler_ale'
)
py.install_sources(
  [
    'src/py_euler_ale/__init__.py',
    'src/py_euler_ale/core.py',
  ],
  subdir: 'py_euler_ale'
)


