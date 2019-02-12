#include "../include/td1.h"
#include <iostream>
#include <time.h>

using namespace std;

// =========================== QUESTION A ==============================

void incr(unsigned int nLoops, volatile double* pCounter)
{
	for (unsigned int iLoop=0; iLoop<nLoops; iLoop++)
	{
		*pCounter = *pCounter + 1.0;
	}
}

void calculate_elapsed(timespec start, timespec end, timespec* elapsed)
{
	elapsed->tv_sec = end.tv_sec - start.tv_sec;
	if (end.tv_nsec < start.tv_nsec)
	{
		elapsed->tv_sec -= 1;
		end.tv_nsec += 1000*1000*1000;
	}
	elapsed->tv_nsec = end.tv_nsec - start.tv_nsec;
}

void timer_handler(int sig, siginfo_t* si, void*)
{
	int *counter = (int*)si->si_value.sival_ptr;
	*counter += 1;
	cout << "Counter: " << *counter << endl;
}

// =========================== QUESTION C ==============================

void incr_stop(int sig, siginfo_t* si, void*)
{
	bool* pStop = (bool*)si->si_value.sival_ptr;
	*pStop = true;
}

unsigned int incr(unsigned int nLoops, double* pCounter, volatile bool* pStop)
{
	for (unsigned int iLoop=0; iLoop<nLoops; iLoop++)
	{
		if (*pStop)
		{
			return iLoop;
		}
		*pCounter = *pCounter + 1.0;
	}
    return nLoops;
}

// =========================== QUESTION E ==============================
double timespec_to_ms(const timespec& time_ts) {
    return time_ts.tv_sec * 1000. + time_ts.tv_nsec/1000000.;
}

timespec timespec_from_ms(double time_ms) {
    struct timespec time_ts;
    time_ts.tv_sec = time_ms/1e3;
    time_ts.tv_nsec = (time_ms-time_ts.tv_sec*1e3)*1e6;
    return time_ts;
}

timespec timespec_now() {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now;
}

timespec timespec_negate(const timespec& time_ts) {
    struct timespec time_neg_ts;
    time_neg_ts.tv_sec = -(time_ts.tv_sec + 1);
    time_neg_ts.tv_nsec = (int) (1e9 - time_ts.tv_nsec);
    return time_neg_ts;
}

timespec timespec_add(const timespec& time1_ts, const timespec& time2_ts) {
    struct timespec timeAdd_ts;
    timeAdd_ts.tv_sec = time1_ts.tv_sec + time2_ts.tv_sec;
    timeAdd_ts.tv_nsec = time1_ts.tv_nsec + time2_ts.tv_nsec;

    if (timeAdd_ts.tv_nsec > 1e9) {
        timeAdd_ts.tv_nsec -= 1e9;
        timeAdd_ts.tv_sec += 1;
    }

    return timeAdd_ts;
}

timespec timespec_subtract(const timespec& time1_ts, const timespec& time2_ts) {
    struct timespec timeSub_ts, time2Neg_ts;

    time2Neg_ts = timespec_negate(time2_ts);
    timeSub_ts = timespec_add(time1_ts, time2Neg_ts);

    return timeSub_ts;
}

timespec timespec_wait(const timespec& delay_ts) {
    timespec rem_ts;
    rem_ts.tv_sec = 0;
    rem_ts.tv_nsec = 0;
    nanosleep(&delay_ts, &rem_ts);
    return rem_ts;
}


timespec  operator- (const timespec& time_ts) {
    return timespec_negate(time_ts);
}

timespec  operator+ (const timespec& time1_ts, const timespec& time2_ts) {
    return timespec_add(time1_ts, time2_ts);
}

timespec  operator- (const timespec& time1_ts, const timespec& time2_ts) {
    return timespec_subtract(time1_ts, time2_ts);
}

timespec& operator+= (timespec& time_ts, const timespec& delay_ts) {
    time_ts = timespec_add(time_ts, delay_ts);
}

timespec& operator-= (timespec& time_ts, const timespec& delay_ts) {
    time_ts = timespec_subtract(time_ts, delay_ts);
}

bool operator== (const timespec& time1_ts, const timespec& time2_ts) {
    return time1_ts.tv_sec == time2_ts.tv_sec && time1_ts.tv_nsec == time2_ts.tv_nsec;
}

bool operator!= (const timespec& time1_ts, const timespec& time2_ts) {
    return time1_ts.tv_sec != time2_ts.tv_sec || time1_ts.tv_nsec != time2_ts.tv_nsec;
}

bool operator< (const timespec& time1_ts, const timespec& time2_ts) {
    if (time1_ts.tv_sec < time2_ts.tv_sec) {
        return true;
    }

    if (time1_ts.tv_sec > time2_ts.tv_sec) {
        return false;
    }

    if (time1_ts.tv_nsec < time2_ts.tv_nsec) {
        return true;
    }

    return false;
}

bool operator> (const timespec& time1_ts, const timespec& time2_ts) {
        if (time1_ts.tv_sec > time2_ts.tv_sec) {
        return true;
    }

    if (time1_ts.tv_sec < time2_ts.tv_sec) {
        return false;
    }

    if (time1_ts.tv_nsec > time2_ts.tv_nsec) {
        return true;
    }

    return false;
}
