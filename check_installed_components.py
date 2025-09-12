# check_installed_components.py
import pkg_resources
import chapter_llama

def check_installation():
    print("=== Chapter-Llama Installation Check ===")
    
    # Check package info
    try:
        dist = pkg_resources.get_distribution("chapter-llama")
        print(f"✅ Package: {dist.project_name} v{dist.version}")
        print(f"Location: {dist.location}")
    except:
        print("❌ Package not found in pkg_resources")
    
    # Check what's in the chapter_llama module
    print("\n📦 Chapter-Llama module contents:")
    for attr in dir(chapter_llama):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Check if we have service components
    print("\n🔍 Checking for service components:")
    try:
        from chapter_llama import server, service, api
        print("✅ Found server/service/api modules!")
    except ImportError as e:
        print(f"❌ Service modules not found: {e}")
    
    # Check entry points
    try:
        dist = pkg_resources.get_distribution("chapter-llama")
        if hasattr(dist, 'get_entry_map'):
            entries = dist.get_entry_map()
            if 'console_scripts' in entries:
                print("🎯 Console scripts available:")
                for name, entry in entries['console_scripts'].items():
                    print(f"  - {name}: {entry.module_name}:{entry.attr_name}")
    except:
        pass

if __name__ == "__main__":
    check_installation()