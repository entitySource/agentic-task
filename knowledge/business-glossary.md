# Business glossary

Shared definitions for the subscription and retail datasets. The agent retrieves
these passages to interpret questions and cites them in its answers.

## Subscription metrics

### Active customer
A customer whose status is not "cancelled". Paused and trial accounts count as
active for reporting, because they can still convert or resume without signing up again.

### Churn and churn rate
A customer has churned when their status is "cancelled". Churn rate for a group is
the number of cancelled customers divided by all customers in that group, shown as a
percentage. Compare churn between plans with a chi-square test, not by eye.

### Monthly recurring revenue (MRR)
MRR is the sum of monthly_revenue across customers. Cancelled customers carry zero
revenue, so MRR already excludes them. Amounts are in Indian rupees (INR).

### Average revenue per account (ARPA)
ARPA is MRR divided by the number of active customers. Use the mean of
monthly_revenue over active customers.

### Seats
The number of licensed users on an account. Basic plans always have one seat.
Revenue per seat is monthly_revenue divided by seats.

### NPS score
The nps column is a 0 to 10 survey answer to "How likely are you to recommend us?".
Scores of 9 or 10 are promoters, 7 or 8 are passives and 0 to 6 are detractors.
Net Promoter Score is the percentage of promoters minus the percentage of detractors.

### Support load
support_tickets counts tickets raised in the last 90 days. More than 4 tickets marks a
high-support account, which the customer success team reviews every week.

### Enterprise contract
An account on the Enterprise plan. Contracts above 500,000 INR a month are negotiated
deals and are expected to look like outliers; check them before excluding any.

## Retail metrics

### Revenue
Revenue for an order is Units multiplied by Unit Price, after the Discount. The
Revenue column already applies the discount.

### Average order value (AOV)
AOV is total Revenue divided by the number of orders (rows). Report it per Store or
per Channel.

### Festive season
The festive season is October to December (Q4). Retail sales normally rise during
this period, so compare Q4 with Q4 of the previous year rather than with Q3.

### Discount depth
Discount is a fraction between 0 and 1. A discount of 0.2 means 20% off. Orders with
a discount of 0.2 or more are deep-discount orders.

### Channel
Online orders come from the website and app; In-store orders come from the five
physical stores (Delhi, Mumbai, Bengaluru, Pune and Jaipur).
