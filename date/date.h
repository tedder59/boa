#ifndef BOA_DATE_DATE_H_
#define BOA_DATE_DATE_H_

class Date
{
public:
    Date(int y, int m, int d);
    ~Date();

    bool isValid() const;
    int year() const;
    int month() const;
    int day() const;
    int dayOfWeek() const;
    int dayOfYear() const;
    int weekOfYear() const;

    int daysTo(const Date&) const;

    static bool isValid(int y, int m, int d);
    static bool isLeapYear(int year);

private:
    int _jds;
};

int operator-(const Date& lhs, const Date& rhs);


#endif // BOA_DATE_DATE_H_