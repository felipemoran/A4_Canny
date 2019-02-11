#ifndef td1_h_INCLUDED
#define td1_h_INCLUDED

#include <time.h>
#include <signal.h>

// =========================== QUESTION A ==============================
void incr(unsigned int nLoops, volatile double* pCounter);
void calculate_elapsed(timespec start, timespec end, timespec* elapsed);
void timer_handler(int sig, siginfo_t* si, void*);

// =========================== QUESTION C ==============================
void incr_stop(int sig, siginfo_t* si, void*);
unsigned int incr(unsigned int nLoops, volatile double* pCounter, volatile bool* pStop);

// =========================== QUESTION E ==============================
double timespec_to_ms(const timespec& time_ts);
timespec timespec_from_ms(double time_ms);
timespec timespec_now();
timespec timespec_negate(const timespec& time_ts);
timespec timespec_add(const timespec& time1_ts, const timespec& time2_ts);
timespec timespec_subtract(const timespec& time1_ts, const timespec& time2_ts);
timespec timespec_wait(const timespec& delay_ts);

timespec  operator- (const timespec& time_ts);
timespec  operator+ (const timespec& time1_ts, const timespec& time2_ts);
timespec  operator- (const timespec& time1_ts, const timespec& time2_ts);
timespec& operator+= (timespec& time_ts, const timespec& delay_ts);
timespec& operator-= (timespec& time_ts, const timespec& delay_ts);
bool operator== (const timespec& time1_ts, const timespec& time2_ts);
bool operator!= (const timespec& time1_ts, const timespec& time2_ts);
bool operator< (const timespec& time1_ts, const timespec& time2_ts);
bool operator> (const timespec& time1_ts, const timespec& time2_ts);

#endif // td1_h_INCLUDED
