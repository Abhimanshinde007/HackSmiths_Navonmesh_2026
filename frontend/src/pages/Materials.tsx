import { useState, useRef } from "react";
import { apiFetch, apiJson } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Upload, Layers, Zap } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function Materials() {
  const [bomData, setBomData] = useState<any[] | null>(null);
  const [forecastData, setForecastData] = useState<any[] | null>(null);
  const [requirements, setRequirements] = useState<any[] | null>(null);
  const [state, setState] = useState<"idle" | "uploading" | "exploding">("idle");
  const bomRef = useRef<HTMLInputElement>(null);
  const forecastRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleBomUpload = async () => {
    const file = bomRef.current?.files?.[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    setState("uploading");
    try {
      const res = await apiFetch("/bom/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      setBomData(json.data);
      toast({ title: "BOM parsed", description: `${json.data.length} items loaded` });
    } catch (err: any) {
      toast({ title: "BOM upload failed", description: err.message, variant: "destructive" });
    } finally {
      setState("idle");
    }
  };

  const handleForecastUpload = async () => {
    const files = forecastRef.current?.files;
    if (!files?.length) return;

    const form = new FormData();
    Array.from(files).forEach(f => form.append("files", f));

    setState("uploading");
    try {
      const res = await apiFetch("/inventory/upload/sales", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      setForecastData(json.data);
      toast({ title: "Forecast data loaded", description: `${json.data.length} records` });
    } catch (err: any) {
      toast({ title: "Upload failed", description: err.message, variant: "destructive" });
    } finally {
      setState("idle");
    }
  };

  const handleExplode = async () => {
    if (!bomData || !forecastData) return;
    setState("exploding");
    try {
      const result = await apiJson<any>("/bom/explode", {
        method: "POST",
        body: JSON.stringify({ bom_data: bomData, forecast_data: forecastData }),
      });
      setRequirements(result.material_requirements);
      toast({ title: "BOM exploded", description: `${result.material_requirements.length} materials computed` });
    } catch (err: any) {
      toast({ title: "Explosion failed", description: err.message, variant: "destructive" });
    } finally {
      setState("idle");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-mono text-2xl font-bold tracking-wider text-foreground">MATERIAL REQUIREMENTS</h1>
        <p className="text-sm text-muted-foreground font-mono">BOM explosion & material planning</p>
      </div>

      {/* BOM Upload */}
      <Card className="border-border/50 bg-card/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
            <Upload className="h-4 w-4" />
            Step 1 — Upload Bill of Materials
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">BOM Excel File</Label>
            <Input ref={bomRef} type="file" accept=".xlsx,.xls" className="bg-muted/50 font-mono text-sm" />
          </div>
          <Button onClick={handleBomUpload} disabled={state === "uploading"} className="font-mono tracking-wider">
            {state === "uploading" ? "PARSING..." : "UPLOAD BOM"}
          </Button>
          {bomData && <p className="text-sm text-primary font-mono">✓ {bomData.length} BOM items loaded</p>}
        </CardContent>
      </Card>

      {/* Forecast Data */}
      <Card className="border-border/50 bg-card/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
            <Layers className="h-4 w-4" />
            Step 2 — Upload Forecast / Sales Data
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Sales Excel Files</Label>
            <Input ref={forecastRef} type="file" multiple accept=".xlsx,.xls" className="bg-muted/50 font-mono text-sm" />
          </div>
          <Button onClick={handleForecastUpload} disabled={state === "uploading"} className="font-mono tracking-wider">
            UPLOAD FORECAST DATA
          </Button>
          {forecastData && <p className="text-sm text-primary font-mono">✓ {forecastData.length} forecast records loaded</p>}
        </CardContent>
      </Card>

      {/* Explode */}
      {bomData && forecastData && (
        <Card className="border-primary/30 bg-card/80 glow-border">
          <CardContent className="flex items-center justify-between pt-6">
            <div>
              <p className="font-mono text-sm text-foreground">Ready to compute material requirements</p>
              <p className="text-xs text-muted-foreground font-mono">{bomData.length} BOM items × {forecastData.length} forecast records</p>
            </div>
            <Button onClick={handleExplode} disabled={state === "exploding"} className="font-mono tracking-wider">
              <Zap className="h-4 w-4 mr-2" />
              {state === "exploding" ? "COMPUTING..." : "EXPLODE BOM"}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {requirements && requirements.length > 0 && (
        <Card className="border-border/50 bg-card/80">
          <CardHeader>
            <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
              Material Requirements ({requirements.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow className="border-border/50">
                  {Object.keys(requirements[0]).map(key => (
                    <TableHead key={key} className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                      {key.replace(/_/g, " ")}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {requirements.map((row, i) => (
                  <TableRow key={i} className="border-border/30">
                    {Object.values(row).map((val: any, j) => (
                      <TableCell key={j} className="font-mono text-sm">
                        {typeof val === "number" ? val.toLocaleString() : String(val)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
