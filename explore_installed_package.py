# explore_installed_package.py
import os
import site
import chapter_llama

def explore_package():
    print("=== Exploring Installed Chapter-Llama Package ===")
    
    # Get the actual installation path
    package_path = os.path.dirname(chapter_llama.__file__)
    print(f"Package path: {package_path}")
    
    # List all files in the package
    print("\n📁 Package contents:")
    for root, dirs, files in os.walk(package_path):
        level = root.replace(package_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f"{subindent}{file}")
    
    # Check if there are submodules
    print("\n🔍 Checking for submodules...")
    try:
        import importlib
        import pkgutil
        
        package = importlib.import_module('chapter_llama')
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
            print(f"Found: {'📦 ' if is_pkg else '📄 '}{name}")
            if is_pkg:
                try:
                    submodule = importlib.import_module(f'chapter_llama.{name}')
                    print(f"  Contents: {[x for x in dir(submodule) if not x.startswith('_')]}")
                except Exception as e:
                    print(f"  Error loading: {e}")
    except Exception as e:
        print(f"Error exploring package: {e}")

if __name__ == "__main__":
    explore_package()