"""
Test script để kiểm tra AI modules hoạt động có đúng không
"""

import sys
import os

def test_ai_import():
    """Test import AI modules"""
    print("🧪 Testing AI Module Imports...")
    
    try:
        # Add parent directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        print(f"📁 Parent directory: {parent_dir}")
        
        # Test DriverMonitor import
        from DriverDetection.driver_monitor import DriverMonitor
        print("✅ DriverMonitor import: OK")
        
        # Test DriverMonitor initialization
        monitor = DriverMonitor()
        print("✅ DriverMonitor init: OK")
        
        # Test model file exists
        model_path = os.path.join(parent_dir, 'models', 'model.keras')
        if os.path.exists(model_path):
            print(f"✅ Model file exists: {model_path}")
        else:
            print(f"❌ Model file missing: {model_path}")
            return False
        
        # Test EyeProcessor
        from DriverDetection.eye_processor import EyeProcessor
        eye_processor = EyeProcessor()
        print("✅ EyeProcessor: OK")
        
        # Test FrameProcessor
        from DriverDetection.frame_processor import FrameProcessor
        frame_processor = FrameProcessor()
        print("✅ FrameProcessor: OK")
        
        print("\n🎉 All AI modules working!")
        return True
        
    except Exception as e:
        print(f"❌ AI modules failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading specifically"""
    print("\n🧪 Testing Model Loading...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        from DriverDetection.model import Models
        models = Models()
        
        # Try to load model
        model = models.load_eye_model()
        print("✅ Eye model loaded successfully")
        print(f"Model type: {type(model)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 AI MODULES DIAGNOSTIC")
    print("=" * 40)
    
    # Test 1: Import modules
    import_ok = test_ai_import()
    
    # Test 2: Model loading
    model_ok = test_model_loading()
    
    print("\n📊 RESULTS:")
    print("=" * 20)
    print(f"AI Imports: {'✅' if import_ok else '❌'}")
    print(f"Model Loading: {'✅' if model_ok else '❌'}")
    
    if import_ok and model_ok:
        print("\n🎉 AI modules ready for main app!")
    else:
        print("\n⚠️ AI modules have issues - app will run in GUI-only mode")
        print("💡 Check if model files exist in ../models/ directory")
