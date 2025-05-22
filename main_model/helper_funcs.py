import xarray as xr
from calliope.backend.helper_functions import ParsingHelperFunction


class SumNextN(ParsingHelperFunction):
    """Sum the next N items in an array. Works best for ordered arrays (datetime, integer)."""

    #:
    NAME = "sum_next_n"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: str, *, over: str, N: int) -> str:  # noqa: D102, override
        overstring = self._instr(over)
        # FIXME: add N
        return rf"\sum\limits_{{{overstring}}} ({array})"

    def as_array(self, array: xr.DataArray, over: str, N: int) -> xr.DataArray:
        """Sum values from current up to N from current on the dimension `over`.

        Args:
            array (xr.DataArray): Math component array.
            over (str): Dimension over which to sum
            N (int): number of items beyond the current value to sum from

        Returns:
            xr.DataArray:
                Returns the input array with the condition applied,
                including having been broadcast across any new dimensions provided by the condition.

        Examples:
            One common use-case is to collate N timesteps beyond a given timestep to apply a constraint to it(e.g., demand must be less than X in the next 24 hours)
        """
        results = []
        for i in range(len(self._input_data.coords[over])):
            results.append(
                array.isel(**{over: slice(i, i + int(N))}).sum(over, min_count=1)
            )
        final_array = xr.concat(
            results, dim=self._input_data.coords[over]
        ).broadcast_like(array)
        return final_array
