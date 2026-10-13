"""Force vLLM pin_memory/UVA available under WSL2 (all Platform subclasses)."""
def apply():
    try:
        import vllm.platforms.interface as iface
        @classmethod
        def _forced(cls) -> bool:
            return True
        iface.Platform.is_pin_memory_available = _forced
        # Patch any already-imported concrete platforms
        try:
            from vllm.platforms import cuda as cuda_mod
            for name in dir(cuda_mod):
                obj = getattr(cuda_mod, name)
                if isinstance(obj, type) and issubclass(obj, iface.Platform):
                    obj.is_pin_memory_available = _forced
        except Exception:
            pass
        try:
            import vllm.utils.platform_utils as pu
            pu.is_pin_memory_available.cache_clear()
            pu.is_uva_available.cache_clear()
        except Exception:
            pass
    except Exception as e:
        print('force_pin_memory apply failed', e)
apply()
