# Data Model Notes

## Level 0 data

- Contain a single 2D image and variance array (2 extensions).
- Image and variance can be empty.
- contain "telemetry" extension as ASCII table (exist in memory as pandas)
- contain "receipt" extension as ASCII table (exist in memory as pandas)
- support adding/removing auxillary HDUs

## Level 1 data

- Data identified by fibers
- Each fiber have flux, wavelength, variance
- contain "telemetry" extension, inherited from Level 0
- contain "receipt" extension, inherited and extended from Level 0
- contains a "segment" extension that specifies all
- inherit all headers keywords from level 0

## Level 2 data

- contain "telemetry" extension, inherited from Level 1
- contain "receipt" extension, inherited and extended from Level 1
- data stored in table. Each row is identified by a segment.
- inherit all headers keywords from level 1

## Demo

### Astropy and FITS

- import astropy, read fits, fits info

### Core

- create empty level 0: info
- receipt: info, access, add, remove
- auxiliary extensions: add, remove

### level 0

- read level 0 NEID
- write level 0 KPF
- read level 0 KPF

### level 1

- read level 1 NEID
- write level 1 KPF
- segement: info, append, delete.
