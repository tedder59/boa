#include "date.h"
#include <assert.h>

static const char monthDays[] = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

Date::Date(int y, int m, int d)
{
}

Date::~Date()
{
}

bool Date::isValid() const
{

}

int Date::year() const
{

}

int Date::month() const
{

}

int Date::day() const
{

}

int Date::dayOfWeek() const
{

}

int Date::dayOfYear() const
{

}

int Date::weekOfYear() const
{

}

int Date::daysTo(const Date&) const
{

}

bool Date::isValid(int y, int m, int d)
{
    if (y <= 0 || y > 9999) return false;

    return (d > 0 && m > 0 && m <= 12) &&
           (d < monthDays[m] || (d == 29 && m == 2 && isLeapYear(y)));
}

bool Date::isLeapYear(int y)
{
    assert(y > 0 && y <= 9999);
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

int operator-(const Date& lhs, const Date& rhs)
{
    
}