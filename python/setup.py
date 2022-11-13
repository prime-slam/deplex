from setuptools import find_packages, setup
import ctypes


cmdclass = dict()

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            if plat[:5] == 'linux':
                libc = ctypes.CDLL('libc.so.6')
                libc.gnu_get_libc_version.restype = ctypes.c_char_p
                GLIBC_VER = libc.gnu_get_libc_version().decode('utf8').split('.')
                plat = f'manylinux_{GLIBC_VER[0]}_{GLIBC_VER[1]}{plat[5:]}'
            return python, abi, plat

    cmdclass['bdist_wheel'] = bdist_wheel

except ImportError:
    print("Warning: wheel package missing!")


with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f.readlines() if line]


setup_args = dict(
    name="deplex",
    version="0.0.1",
    install_requires=install_requires,
    packges=find_packages(),
    cmdclass=cmdclass
)

setup(**setup_args)
