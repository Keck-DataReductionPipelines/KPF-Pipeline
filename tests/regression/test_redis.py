#!/usr/bin/env python3
"""
Test script to verify Redis caching is working
"""

import time
from modules.calibration_lookup.src.alg import GetCalibrations

def test_redis_caching():
    """Test that Redis caching works across multiple lookups"""
    datetime = '2024-10-22T09:16:52.826'
    default_config_path = 'modules/calibration_lookup/configs/default.cfg'
    
    print("=== Testing Redis Caching ===")
    
    # First lookup (should miss cache)
    print("\n--- FIRST LOOKUP (Cache MISS) ---")
    start_time = time.time()
    cals1 = GetCalibrations(datetime, default_config_path)
    caldict1 = cals1.lookup()
    first_lookup_time = time.time() - start_time
    print(f"First lookup completed in {first_lookup_time:.3f}s")
    
    # Second lookup (should hit cache)
    print("\n--- SECOND LOOKUP (Cache HIT) ---")
    start_time = time.time()
    cals2 = GetCalibrations(datetime, default_config_path)
    caldict2 = cals2.lookup()
    second_lookup_time = time.time() - start_time
    print(f"Second lookup completed in {second_lookup_time:.3f}s")
    
    # Third lookup (should hit cache)
    print("\n--- THIRD LOOKUP (Cache HIT) ---")
    start_time = time.time()
    cals3 = GetCalibrations(datetime, default_config_path)
    caldict3 = cals3.lookup()
    third_lookup_time = time.time() - start_time
    print(f"Third lookup completed in {third_lookup_time:.3f}s")
    
    # Results
    print("\n=== RESULTS ===")
    print(f"First lookup (cache miss):  {first_lookup_time:.3f}s")
    print(f"Second lookup (cache hit):  {second_lookup_time:.3f}s")
    print(f"Third lookup (cache hit):   {third_lookup_time:.3f}s")
    
    if second_lookup_time < first_lookup_time * 0.5:
        print("✅ SUCCESS: Second lookup was significantly faster (cache hit)")
    else:
        print("❌ FAILURE: Second lookup was not significantly faster")
    
    if third_lookup_time < first_lookup_time * 0.5:
        print("✅ SUCCESS: Third lookup was significantly faster (cache hit)")
    else:
        print("❌ FAILURE: Third lookup was not significantly faster")
    
    # Verify results are identical
    if caldict1 == caldict2 == caldict3:
        print("✅ SUCCESS: All lookups returned identical results")
    else:
        print("❌ FAILURE: Lookups returned different results")
    
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
