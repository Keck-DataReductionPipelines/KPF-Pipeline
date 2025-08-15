import logging
import sys
import time


from modules.calibration_lookup.src.alg import GetCalibrations
from modules.calibration_lookup.src.calibration_lookup import CalibrationLookup


def test_calibration_lookup_caching():
    """Test that Redis caching works across multiple lookups"""
    # datetime = '2024-10-22T09:16:52.826'
    datetime = '2024-02-28T09:16:52.826'
    default_config_path = 'modules/calibration_lookup/configs/default.cfg'
    
    print("=== Testing Redis Caching ===")
    
    # First lookup (cache miss) - only get a subset of keys
    print("\n--- FIRST LOOKUP (Cache MISS - Partial) ---")
    start_time = time.time()
    cals1 = GetCalibrations(datetime, default_config_path)
    # Only request some keys initially, not all
    initial_keys = ['bias', 'dark', 'flat', 'wls', 'rough_wls']
    caldict1 = cals1.lookup(subset=initial_keys)
    first_lookup_time = time.time() - start_time
    print(f"First lookup (subset: {initial_keys}) completed in {first_lookup_time:.3f}s")
    print(f"Retrieved {len(caldict1)} calibration types")
    
    # Second lookup (cache hit) - same subset
    print("\n--- SECOND LOOKUP (Cache HIT - Same subset) ---")
    start_time = time.time()
    cals2 = GetCalibrations(datetime, default_config_path)
    caldict2 = cals2.lookup(subset=initial_keys)
    second_lookup_time = time.time() - start_time
    print(f"Second lookup (subset: {initial_keys}) completed in {second_lookup_time:.3f}s")
    print(f"Retrieved {len(caldict2)} calibration types")
    
    # Third lookup (partial cache hit using expanded subset)
    print("\n--- THIRD LOOKUP (Partial Cache HIT with expanded subset) ---")
    start_time = time.time()
    cals3 = GetCalibrations(datetime, default_config_path)
    # Request keys that include some cached + some new ones
    expanded_keys = ['bias', 'dark', 'flat', 'wls', 'rough_wls', 'ordertrace', 'start_order', 'etalon_drift']
    # 5 cached keys + 3 new keys = 8 total
    caldict3 = cals3.lookup(subset=expanded_keys)
    third_lookup_time = time.time() - start_time
    print(f"Third lookup (subset: {expanded_keys}) completed in {third_lookup_time:.3f}s")
    print(f"Retrieved {len(caldict3)} calibration types (expected: {len(expanded_keys)})")
    
    # Results
    print("\n=== RESULTS ===")
    print(f"First lookup (cache miss):  {first_lookup_time:.3f}s")
    print(f"Second lookup (cache hit):  {second_lookup_time:.3f}s")
    print(f"Third lookup (partial cache hit):   {third_lookup_time:.3f}s")
    
    if second_lookup_time < first_lookup_time * 0.5:
        print("✅ SUCCESS: Second lookup was significantly faster (cache hit)")
    else:
        print("❌ FAILURE: Second lookup was not significantly faster")
    
    if third_lookup_time < first_lookup_time * 0.5:
        print("✅ SUCCESS: Third lookup was significantly faster (partial cache hit)")
    else:
        print("❌ FAILURE: Third lookup was not significantly faster")
    
    # Verify results are consistent
    # First two lookups should be identical (same subset cache hit)
    if caldict1 == caldict2:
        print("✅ SUCCESS: First two lookups returned identical results (same subset cache hit)")
    else:
        print("❌ FAILURE: First two lookups returned different results")
    
    # Third lookup should only contain the requested expanded subset keys
    if set(caldict3.keys()) == set(expanded_keys):
        print("✅ SUCCESS: Third lookup returned exactly the requested expanded subset keys")
    else:
        print("❌ FAILURE: Third lookup did not return the requested expanded subset keys")
    
    # Clear cache at the end for clean test state
    print("\n--- Clearing Cache ---")
    try:
        from database.modules.utils.kpf_db import clear_cache
        clear_cache()
        print("✅ Cache cleared successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not clear cache: {e}")
    
    print("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_redis_caching()


if __name__ == '__main__':
    test_calibration_lookup()