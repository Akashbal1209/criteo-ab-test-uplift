import { createContext, useContext } from "react";

// true = technical view (statistical terms), false = plain-language view
export const TechContext = createContext(false);
export const useTech = () => useContext(TechContext);
