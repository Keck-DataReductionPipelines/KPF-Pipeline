# Data Model Notes

## Level 0 data

- Contain a single 2D image and variance array (2 extensions).
- Image and variance can be empty.
- contain "receipt" extension as ASCII table (exist in memory as pandas)
- support adding/removing auxillary HDUs

## Level 1 data

- Data identified by fibers
- Each fiber have flux, wavelength, variance
- contain "receipt" extension
- contains a "segment" extension that specifies all
- inherit all headers keywords from level 0

## Level 2 data

- data stored in table. Each row is identified by a segment.