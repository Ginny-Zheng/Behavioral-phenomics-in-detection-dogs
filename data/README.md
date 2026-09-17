# Data schema

Place the confidential input workbook at `data/raw/Data_with_CST.xlsx`. Do not commit this file.

Each row represents one dog. The required fields are:

- `DogID`
- `final_disposition`, with values `Sale Quality` or `Washout`
- Eleven phenotype columns at each of 3, 6, 10, and 12 months, named `<phenotype>.Time<age>`
- `CST.Time3`, `CST.Time6`, `CST.Time10`, and `CST.Time12` for the CSS comparisons

The phenotype stems are `Retrieve`, `Hunt`, `Focus.on.Toy/Reward`, `Physical.Possessiveness.of.Toy`, `Independence`, `Work/Effort`, `Air.Scenting`, `Surfaces`, `People`, `Vehicles.&.Urban.Clutter`, and `Excitability`.

The analysis dataset contains 180 dogs, 115 classified as Sale Quality and 65 classified as Washout, with complete phenotype scores at all four timepoints.
