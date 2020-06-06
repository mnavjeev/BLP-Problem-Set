set more off
cd "/Users/manunavjeevan/Desktop/UCLA/Second Year/Winter 2020/IO/Problem Set 1"
*import delimited using dataTotal.csv, delim(",") clear
*drop v1 
*save dataTotal_raw, replace
use dataTotal_raw, clear
order market, after(week)
rename sales_ sales
rename price_ price
rename cost_ cost
rename prom_ prom
gen share = sales/count


order share, after(brand)
order leaveoutav, after(price)
rename leaveoutav leaveOutPrice

local pricestores
foreach i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 {
 rename pricestore`i' pricestore`i'_
 local pricestores `pricestores' pricestore`i'_
}
drop avoutprice


reshape wide sales price prom cost share leaveOutPrice `pricestores', i(market) j(brand)
order store week count, after(market)

order share2 share3 share4 share5 share6 share7 share8 share9 share10 share11, after(share1)
gen share12 = 1 - share2 - share3 - share4 - share5 - share6 - share7 - share8 - share9 - share10 - share11
order share12, after(share11)

local shareDiffs
foreach i in 1 2 3 4 5 6 7 8 9 10 11 {
	gen shareDiff`i' = ln(share`i') - ln(share12)
	local shareDiffs `shareDiffs' shareDiff`i'
}
order `shareDiffs', after(share12)

reshape long
gen shareDiff = .
order shareDiff, after(share)
foreach i in 1 2 3 4 5 6 7 8 9 10 11 {
	replace shareDiff = shareDiff`i' if brand == `i'
}

drop if brand == 12
drop `shareDiffs'
foreach i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 {
 rename pricestore`i'_ pricestore`i'
 replace pricestore`i' = 0 if store == `i'
}
gen branded = 1 if brand <= 9
replace branded = 0 if branded != 1


export delimited using "dataCleaned2.csv", delim(",") replace

