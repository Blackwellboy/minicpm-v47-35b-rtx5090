"""Native-transformers overlay for MiniCPM-V-4.7-35B-A3B-AWQ.

The AWQ repo (ea491d0) still ships auto_map pointers to processing_minicpmv4_7.py etc. that are not in the repo,
so transformers demands trust_remote_code and then fails. The BF16 base repo (17e6511, 2026-10-13) moved to the
native transformers>=5.18 configs. Symlink the AWQ weights and take the three processor/tokenizer configs from base.
"""
import os, shutil, sys
awq, base, dst = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
donor = {"processor_config.json", "preprocessor_config.json", "tokenizer_config.json"}
for f in os.listdir(awq):
    d = os.path.join(dst, f)
    if os.path.lexists(d): os.remove(d)
    if f in donor:
        shutil.copyfile(os.path.join(base, f), d); print("from base:", f)
    else:
        os.symlink(os.path.join(awq, f), d)
