

# Industrial Supply Chain Dashboard

## Overview
A premium industrial dashboard connecting to a local backend at `http://localhost:8001`, featuring authentication, inventory management, demand forecasting, and material requirements planning. Deep blue & steel grey color scheme with a techy, industrial aesthetic.

## Design System
- **Primary**: Deep navy blue (`hsl(220, 60%, 15%)`)
- **Accents**: Electric blue highlights, steel grey surfaces
- **Style**: Dark-mode industrial feel — sharp edges, monospace accents, glowing borders, data-dense layouts
- **Charts**: Recharts with electric blue/cyan/steel color palette

## Pages & Features

### 1. Login Page (`/login`)
- Username + password form (sends `application/x-www-form-urlencoded` to `POST /auth/login`)
- Stores JWT in localStorage + React Context auth provider
- Auto-redirects to dashboard on success
- Protected route wrapper redirects unauthenticated users to `/login`

### 2. App Layout
- Collapsible sidebar with icons: Inventory, Forecast, Materials
- Top header bar with user info (from `GET /auth/me`) and logout button
- SidebarProvider + SidebarTrigger always visible

### 3. Inventory Stock Page (`/inventory`)
- Fetches `GET /inventory/stock` on mount
- Displays data in a sortable table (SKU, Name, Current Stock, Safety Stock)
- Color-coded rows: red when current_stock < safety_stock
- Bar chart (Recharts) comparing current vs safety stock levels
- File upload section for inward/outward Excel files (`POST /inventory/upload/stock`)

### 4. Demand Forecast Page (`/forecast`)
- Step 1: Upload sales Excel files (`POST /inventory/upload/sales`) — returns parsed sales data
- Step 2: Set prediction window (default 90 days) and run forecast (`POST /forecast/run`)
- Results: Display anchor customers list, forecast line chart, and any errors
- Recharts line/area chart for forecast visualization

### 5. Material Requirements Page (`/materials`)
- Step 1: Upload BOM file (`POST /bom/upload`) — returns parsed BOM data
- Step 2: Select/confirm forecast data, then explode (`POST /bom/explode`)
- Results: Table of material requirements with quantities

## Technical Architecture
- **Auth Context**: Stores token + user info, provides `login()`, `logout()`, `isAuthenticated`
- **API utility**: Central fetch wrapper with base URL `http://localhost:8001`, auto-attaches Bearer token
- **Protected Routes**: Wrapper component checks auth, redirects to `/login`
- **No mock data**: All data comes from the real backend API

