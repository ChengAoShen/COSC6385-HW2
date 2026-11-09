#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>
int main(){
  uint8_t buf[4096]; volatile uint8_t *p=&buf[1024]; unsigned junk;
  uint64_t hit=0,miss=0; int N=200000;
  for(int i=0;i<N;i++){ // cached
    _mm_clflush((void*)p); for(volatile int d=0;d<1000;d++); *p; // bring in
    uint64_t t=__rdtscp(&junk); junk+=*p; uint64_t e=__rdtscp(&junk)-t; hit+=e;
  }
  for(int i=0;i<N;i++){ // flushed
    _mm_clflush((void*)p); _mm_mfence();
    uint64_t t=__rdtscp(&junk); junk+=*p; uint64_t e=__rdtscp(&junk)-t; miss+=e;
  }
  printf("avg cache HIT : %.1f cycles\navg cache MISS: %.1f cycles\n",(double)hit/N,(double)miss/N);
  return junk;
}
